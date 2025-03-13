import sys
import os
from torch.utils.data import DataLoader
import traceback
from tqdm import tqdm
import json 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import torch

# 配置部分
dataset_name = 'iemocap'  # 假设新的中文数据集仍命名为iemoCap
data_folder = './cndata/'
prompt_type = 'spdescV2'

print("Loading model ...")
# 使用支持中文的Qwen模型
model_name = '/apdcephfs_cq10/share_1615176/cq2/macy/BiosERC/llm_model/Qwen2.5-7B-Instruct'  # 请替换为实际的Qwen模型路径或名称
tensor_data_type = torch.bfloat16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=tensor_data_type
)

# 加载预训练的Qwen模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    # low_cpu_mem_usage=True,  # 可选，根据需要启用
)

# 加载对应的分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

class BatchPreprocessor(object): 
    def __init__(self, tokenizer, dataset_name=None, window_ct=2) -> None:
        self.tokenizer = tokenizer
        self.separate_token_id = self.tokenizer.convert_tokens_to_ids("</s>")
        self.dataset_name  = dataset_name
        self.window_ct = window_ct
    
    @staticmethod
    def load_raw_data(path_data):
        with open(path_data, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        if isinstance(raw_data, dict):
            new_data_list = []
            for k, v in raw_data.items():
                v['s_id'] = k
                new_data_list.append(v)
            return new_data_list
        elif isinstance(raw_data, list):
            return raw_data
                
    def sentence_mixed_by_surrounding(self, sentences, around_window, speakers):
        new_sentences = []
        for i, cur_sent in enumerate(sentences):
            tmp_s = ""
            for j in range(max(0, i-around_window), min(len(sentences), i+around_window+1)):
                if i == j:
                    tmp_s += " </s>"
                tmp_s +=  f" {speakers[j]}: {sentences[j]}"
                if i == j:
                    tmp_s += " </s>"
            new_sentences.append(tmp_s)
        return new_sentences
    
    def __call__(self, batch):
        raw_sentences = []
        raw_sentences_flatten = []
        labels = []

        # masked tensor  
        lengths = [len(sample['sentences']) for sample in batch]
        max_len_conversation = max(lengths)
        padding_utterance_masked = torch.BoolTensor([[False]*l_i + [True]*(max_len_conversation - l_i) for l_i in lengths])

        # collect all sentences
        # - intra speaker
        intra_speaker_masekd_all = torch.BoolTensor(len(batch), max_len_conversation, max_len_conversation)
        for i, sample in enumerate(batch):
            sentences_mixed_arround = self.sentence_mixed_by_surrounding(
                sample['sentences'], 
                around_window=self.window_ct, 
                speakers=sample['speakers']
            )
        
            # conversation padding 
            padded_conversation = sentences_mixed_arround + ["<pad_sentence>"] * (max_len_conversation - lengths[i])
            raw_sentences.append(padded_conversation)
            raw_sentences_flatten += padded_conversation

            # label padding 
            labels += [int(label) for label in sample['labels']] + [-1] * (max_len_conversation - lengths[i])

            # speaker mask
            intra_speaker_masekd = torch.BoolTensor(len(padded_conversation), len(padded_conversation)).fill_(False)
            for j in range(len(sample['genders'])):
                for k in range(len(sample['genders'])):
                    gender_j = sample['genders'][j]
                    gender_k = sample['genders'][k]

                    if gender_j == gender_k:
                        intra_speaker_masekd[j][k] = True
                    else:
                        intra_speaker_masekd[j][k] = False

            intra_speaker_masekd_all[i] = intra_speaker_masekd

        if len(labels) != len(raw_sentences_flatten):
            print('len(labels)!= len(raw_sentences_flatten)')

        # utterance vectorizer
        contextual_sentences_ids = self.tokenizer(
            raw_sentences_flatten,  
            padding='longest', 
            max_length=512, 
            truncation=True, 
            return_tensors='pt'
        )
        sent_indices, word_indices = torch.where(contextual_sentences_ids['input_ids'] == self.separate_token_id)
        gr_sent_indices = [[] for _ in range(len(raw_sentences_flatten))]
        for sent_idx, w_idx in zip(sent_indices, word_indices):
            gr_sent_indices[sent_idx].append(w_idx.item())
            
        cur_sentence_indexes_masked = torch.BoolTensor(contextual_sentences_ids['input_ids'].shape).fill_(False)
        for i in range(contextual_sentences_ids['input_ids'].shape[0]):
            if raw_sentences_flatten[i] == '<pad_sentence>':
                cur_sentence_indexes_masked[i][gr_sent_indices[i][0]] = True
                continue
            for j in range(contextual_sentences_ids['input_ids'].shape[1]):
                if  gr_sent_indices[i][0] <= j <= gr_sent_indices[i][1]:
                    cur_sentence_indexes_masked[i][j] = True

        return (contextual_sentences_ids, torch.LongTensor(labels), padding_utterance_masked, intra_speaker_masekd_all, cur_sentence_indexes_masked, raw_sentences) 

class BatchPreprocessorLLM(BatchPreprocessor):
    def __init__(self, tokenizer, dataset_name=None, window_ct=2, emotion_labels=[]) -> None:
        super().__init__(tokenizer, dataset_name, window_ct)
        self.emotion_labels = emotion_labels
        self.printted = False

    def sentence_mixed_by_surrounding(self, sentences, around_window, speakers):
        new_conversations = []
        align_sents = []
        for i, cur_sent in enumerate(sentences):
            tmp_s = ""
            for j in range(max(0, i-around_window), min(len(sentences), i+around_window+1)):
                u_j = f"{speakers[j]}: {sentences[j]}"
                if i == j:
                    align_sents.append(u_j)
                tmp_s += f"\n{u_j}"
            new_conversations.append(tmp_s)
        return new_conversations, align_sents

    def __call__(self, batch):
        flatten_data = []
        few_shot_example = """\n=======
上下文：给定预定义的情感标签集 [快乐, 悲伤, 中性, 愤怒, 兴奋, 挫败]，以及以下对话内容：
"
张晓红: 你好，这里的空气真好，味道很甜。
张晓红: 不，不是抱歉。但，我不会留下来的。
王凯: 问题是，我原本计划在大约一周的时间里偷偷接近你。但他们理所当然地认为我们都已经准备好了。
张晓红: 我知道他们会的，你的母亲也是。
张晓红: 好吧，从她的角度来看，我为什么还要来？
张晓红: 我猜这就是我来的原因。
王凯: 我让你难堪，我不想在这里告诉你。我想去一个我们从未去过的地方，一个我们彼此全新的地方。
张晓红: 好吧，你开始写信给我了。
王凯: 你这么早就感觉到了？
张晓红: 从那天起每天都是。
王凯: 思齐，你为什么不让我知道？
王凯: 我们开车去别的地方吧。我想和你独处。
王凯: 不，没什么那样的。
"

问题：在句子 "张晓红: 我猜这就是我来的原因。" 中，讲话人的情感是什么？
答案：悲伤

问题：在句子 "王凯: 我让你难堪，我不想在这里告诉你。我想去一个我们从未去过的地方，一个我们彼此全新的地方。" 中，讲话人的情感是什么？
答案：兴奋

问题：在句子 "张晓红: 你好，这里的空气真好，味道很甜。" 中，讲话人的情感是什么？
答案：快乐
"""

        for sample in batch:
            new_conversations, align_sents = self.sentence_mixed_by_surrounding(
                sample['sentences'],
                around_window=self.window_ct,
                speakers=sample['speakers']
            )
            for conv, utterance in zip(new_conversations, align_sents):
                prompt_extract_context_vect = few_shot_example + \
                    f"\n=======\n上下文：给定预定义的情感标签集 [{', '.join(self.emotion_labels)}]，以及以下对话内容：\n\"{conv}\n\"\n\n问题：在句子 \"{utterance}\" 中，讲话人的情感是什么？\n答案："
                if not self.printted:
                    print(prompt_extract_context_vect)
                    self.printted = True

                inputs = self.tokenizer(
                    prompt_extract_context_vect, 
                    return_tensors="pt"
                )
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]  # 显式设置注意力掩码
                flatten_data.append({
                    "s_id": sample['s_id'],
                    "prompt_content": prompt_extract_context_vect,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                })

        return flatten_data


class BatchPreprocessorLLMSpeakerDescription(BatchPreprocessor):
    def __init__(self, tokenizer, dataset_name=None, window_ct=2, emotion_labels=[]) -> None:
        super().__init__(tokenizer, dataset_name, window_ct)
        self.emotion_labels = emotion_labels

    def preprocess(self, all_conversations):
        gr_by_len = {}
        for sample in all_conversations:
            all_utterances = []
            all_speaker_names = []
            for idx, u in enumerate(sample['sentences']):
                speaker_name = sample['speakers'][idx]
                sentence = u.strip()
                u_full_name = f'{speaker_name}: {sentence}'
                all_utterances.append(u_full_name)
                all_speaker_names.append(speaker_name)

            full_conversation = "\n".join(all_utterances)
            prompting_input = {}
            for speaker_name in set(all_speaker_names):
                prompting = f"\n给定以下说话人之间的对话：\n\"{full_conversation}\"\n\n请描述说话人 {speaker_name} 的特征。（请在250字以内回答）"
                prompting_input[speaker_name] = prompting

                # 分组按提示长度
                input_ids = self.tokenizer(prompting, return_tensors="pt")["input_ids"]
                prompt_length = input_ids.shape[-1]
                if prompt_length not in gr_by_len:
                    gr_by_len[prompt_length] = []
                gr_by_len[prompt_length].append({
                    'w_ids': input_ids,
                    'conv_id': sample['s_id'],
                    'type_data': sample['type_data'],
                    "prompting_input": prompting,
                    'speaker_name': speaker_name,
                    'all_speaker_names': all_speaker_names
                })

        return gr_by_len

# 加载和处理原始数据
raw_data = []
for type_data in ['valid', 'test', 'train']:
    data_name_pattern = f'{dataset_name}.{type_data}'
    path_processed_data = f'{data_folder}/{data_name_pattern}_{prompt_type}_{model_name.split("/")[-1]}.json'

    org_raw_data = BatchPreprocessorLLMSpeakerDescription.load_raw_data(
        f"{data_folder}/{data_name_pattern}.json"
    )

    if os.path.exists(path_processed_data):
        with open(path_processed_data, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        print(f'- 成功加载已处理的 {len(processed_data)}/{len(org_raw_data)} 个会话，数据类型={type_data}')
        with open(path_processed_data + "_backup.json", 'w', encoding='utf-8') as f_backup:
            json.dump(processed_data, f_backup, ensure_ascii=False, indent=2)
        org_raw_data = [e for e in org_raw_data if e['s_id'] not in processed_data]

    print(f'- 继续处理 {len(org_raw_data)} 个会话，数据类型={type_data}')
    for e in org_raw_data:
        e['type_data'] = type_data
    raw_data += org_raw_data

# 初始化数据预处理器
data_preprocessor = BatchPreprocessorLLMSpeakerDescription(
    tokenizer, 
    dataset_name=dataset_name, 
    window_ct=4,
    emotion_labels=['快乐', '悲伤', '中性', '愤怒', '兴奋', '挫败']
)

gr_by_len = data_preprocessor.preprocess(raw_data)
all_data = {}
print_one_time = True

for len_promting, speaker_promts in tqdm(gr_by_len.items()):
    for batch_size in [32, 16 ,8]:
        try:
            all_promtings_texts = [e['prompting_input'] for e in speaker_promts]
            data_loader = DataLoader(all_promtings_texts, batch_size=batch_size, shuffle=False)
            output_sp_desc = []
            with torch.no_grad():
                for speaker_promts_in_batch in data_loader:
                    inputs = tokenizer(speaker_promts_in_batch, return_tensors="pt", padding=False)
                    input_ids = inputs["input_ids"].to("cuda")
                    attention_mask = inputs["attention_mask"].to("cuda")  # 显式设置注意力掩码
                    with torch.no_grad():
                        outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=300)
                    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                    for j, e in enumerate(output_text):
                        output_sp_desc.append(e.replace(all_promtings_texts[j], "").strip())

                    if print_one_time:
                        print(output_text)
                        print(output_sp_desc)
                        print_one_time = False

                for i, out in enumerate(output_sp_desc):
                    speaker_promts[i]['sp_desc'] = out
            break

        except Exception as e:
            traceback.print_exc()
            print(e)
            if batch_size == 1:
                print(["Errr "] * 10)

# 保存处理后的数据
for type_data in ['valid', 'test', 'train']:
    data_name_pattern = f'{dataset_name}.{type_data}'
    path_processed_data = f'{data_folder}/{data_name_pattern}_{prompt_type}_{model_name.split("/")[-1]}.json'

    processed_data = {}
    if os.path.exists(path_processed_data):
        with open(path_processed_data, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        print(f'- 加载已处理的 [旧] {len(processed_data)} 个会话，数据类型={type_data}')

    all_data = {}
    for len_promting, speaker_promts in gr_by_len.items():
        for description in speaker_promts:
            if type_data != description['type_data']:
                continue

            if description['conv_id'] not in all_data:
                all_data[description['conv_id']] = {
                    'all_speaker_names': description['all_speaker_names'],
                    'vocab_sp2desc':  {}
                }
            all_data[description['conv_id']]['vocab_sp2desc'][description['speaker_name']] = description['sp_desc']

    print(f'- 成功处理 [新] {len(all_data)} 个会话，数据类型={type_data}')

    all_data_new = {}
    for k, v in all_data.items():
        all_data_new[k] = []
        for sp_name in v['all_speaker_names']:
            all_data_new[k].append(v['vocab_sp2desc'][sp_name])

    print(f'- 更新已处理的数据 [新] {len(all_data_new)} + [旧] {len(processed_data)} 个会话，数据类型={type_data}')
    all_data_new.update(processed_data)
    with open(path_processed_data, 'w', encoding='utf-8') as f_out:
        json.dump(all_data_new, f_out, ensure_ascii=False, indent=2)
