import argparse
import json
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import setup_chat_format, set_seed as trl_seed
from sklearn.metrics import f1_score
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from lightning import seed_everything

# 假设您有一个与微调脚本中相同的 `process` 函数用于数据预处理
from reformat_data_ft_llm import process

def set_random_seed(seed: int):
    """设置随机种子以确保结果可复现"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_label(sample, tokenizer):
    """处理标签，将最后一条消息编码为标签"""
    tokenized_lb = tokenizer.encode(
        sample['messages'][-1]['content'], 
        padding='max_length', 
        max_length=10, 
        truncation=True
    )
    sample['labels'] = tokenized_lb
    return sample

def post_process(str_out):
    """后处理生成的文本，提取助手的回答部分"""
    try:
        gen_text = str_out.split("assistant\n")[-1].split("<|im_end|>")[0]
    except:
        gen_text = "error"
    return gen_text

def custom_collate_fn(batch):
    """自定义的collate_fn，确保批次中只包含input_ids, attention_mask, labels，并将它们转换为张量"""
    input_ids = torch.tensor([sample['input_ids'] for sample in batch], dtype=torch.long)
    attention_mask = torch.tensor([sample['attention_mask'] for sample in batch], dtype=torch.long)
    labels = torch.tensor([sample['labels'] for sample in batch], dtype=torch.long)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def evaluate(model, tokenizer, dataloader, device):
    """评估模型并计算加权F1分数"""
    model.eval()
    all_preds = []
    all_labels = []
    all_raw_decoded = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 提取 input_ids, attention_mask, labels
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']  # 保持为列表

            gen_kwargs = {
                'max_new_tokens': 10,
                'do_sample': False,
                'eos_token_id': tokenizer.eos_token_id,
                'pad_token_id': tokenizer.pad_token_id,
                "temperature": 0.1,
            }
            generated_tokens = model.generate(
                input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
            # 解码标签和预测
            str_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            raw_decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
            str_decoded = [post_process(e) for e in raw_decoded]
            all_preds += str_decoded
            all_labels += str_labels
            all_raw_decoded += raw_decoded

    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    return f1_weighted, list(zip(all_preds, all_labels, all_raw_decoded))

def main(args):
    # 设置随机种子
    set_random_seed(args.seed)

    # 处理数据（如果需要）
    all_path_folder_preprocessed_data = [
        f"{args.data_folder}/{args.data_name}.{d_type}.{args.kshot}shot_w{args.window}_{args.prompting_type}.jsonl"
        for d_type in ['train', 'valid', 'test']
    ]
    if args.re_gen_data:
        process(all_path_folder_preprocessed_data, args)

    # 加载测试集
    test_dataset = load_dataset(
        "json",
        data_files=all_path_folder_preprocessed_data[2],
        split="train",
        cache_dir=f'{args.output_folder}/'
    )

    # 加载基座模型和分词器
    base_model_id = args.base_model_id
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.float32  # 根据需要调整数据类型
    )
    model, tokenizer = setup_chat_format(model, tokenizer)
    model.to(args.device)

    # 预处理测试集
    def preprocess_function(sample):
        sample = split_label(sample, tokenizer)
        return sample

    test_dataset = test_dataset.map(preprocess_function, batched=False)

    # Tokenize the dataset
    def tokenize_function(sample):
        # 应用聊天模板到 messages[:-1]
        # 确保 apply_chat_template 返回一个字符串
        prompt_text = tokenizer.apply_chat_template(
            sample['messages'][:-1], 
            tokenize=False, 
            add_generation_prompt=True
        )
        tokenized = tokenizer(
            prompt_text, 
            padding='max_length', 
            truncation=True, 
            max_length=512
        )
        # 分配标记化的字段
        sample['input_ids'] = tokenized['input_ids']
        sample['attention_mask'] = tokenized['attention_mask']
        # 移除 'messages' 以防止 data collator 处理它
        del sample['messages']
        return sample

    test_dataset = test_dataset.map(tokenize_function, batched=False)

    # 确保数据集中只包含 'input_ids', 'attention_mask', 'labels'
    test_dataset = test_dataset.remove_columns([col for col in test_dataset.column_names if col not in ['input_ids', 'attention_mask', 'labels']])

    # 创建DataLoader，使用自定义的collate_fn确保批处理时数据格式正确
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn  # 使用自定义的collate_fn
    )

    # 执行评估
    f1, details = evaluate(model, tokenizer, test_dataloader, args.device)
    print(f"Base Model Test Weighted F1 Score: {f1}")

    # 可选：保存详细预测结果
    if args.save_details:
        with open(args.details_output_path, "w") as f:
            json.dump({"f1_weighted": f1, "detail_pred": details}, f, indent=2)
        print(f"Detailed predictions saved to {args.details_output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Base Llama Model F1 Score')
    parser.add_argument('--base_model_id', type=str, default='/apdcephfs_cq10/share_1615176/cq2/macy/BiosERC/llm_model/llama3.2-3b', help='Base LLM model id')
    # 移除 ft_model_id 参数，因为不需要微调模型的ID
    parser.add_argument('--data_name', type=str, help='Data name in {iemocap, meld, emorynlp}', default='iemocap')
    parser.add_argument('--data_folder', type=str, help='Path to the data folder', default='./data/')
    parser.add_argument('--output_folder', type=str, help='Path to save outputs', default='./finetuned_llm/')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1, help='Evaluation batch size per device')
    parser.add_argument('--prompting_type', type=str, default='spdescV2', help='Prompting style in {cot, fewshot, zeroshot}')
    parser.add_argument('--kshot', type=int, default=0, help='k-shot examples for LLM')
    parser.add_argument('--window', type=int, default=5, help='Local context window size')
    parser.add_argument('--re_gen_data', action="store_true", help='Re-generate data', default=False)
    parser.add_argument('--save_details', action="store_true", help='Save detailed predictions', default=False)
    parser.add_argument('--details_output_path', type=str, default='base_model_evaluation_details.json', help='Path to save detailed predictions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed value')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(args)
