import json
import re

def get_speaker_name(s_id, gender, data_name, speaker=None):
    """
    获取说话人名称。
    如果提供了 speaker 参数，则直接返回该名称（适用于中文数据）。
    否则，根据 s_id 和 gender 进行映射（适用于英文数据）。
    """
    if speaker is not None:
        return speaker
    if data_name == "iemocap":
        # iemocap: {'Happy':0, 'Neutral':1, 'Sad':2, 'Disgust':3, 'Anger':4, 'Fear':5, 'Surprise':6}
        speaker_mapping = {
            "Ses01": {"F": "张晓红", "M": "王凯"},
            "Ses02": {"F": "李丽", "M": "刘伟"},
            "Ses03": {"F": "赵敏", "M": "陈强"},
            "Ses04": {"F": "孙婷", "M": "周杰"},
            "Ses05": {"F": "吴静", "M": "郑宇"},
        }
        s_id_first_part = s_id[:5]
        if s_id_first_part in speaker_mapping:
            return speaker_mapping[s_id_first_part][gender]
        else:
            raise KeyError(f"Speaker ID prefix '{s_id_first_part}' not found in speaker mapping.")
    elif data_name in ['meld', "emorynlp"]:
        # emorynlp: {'Happy':0, 'Mad':1, 'Peaceful':2, 'Neutral':3, 'Sad':4, 'Powerful':5, 'Scared':6}
        # meld: {'neutral':0, 'surprise':1, 'fear':2, 'sadness':3, 'joy':4, 'disgust':5, 'anger':6}
        gender_idx = gender.index(1)
        return f"说话人_{gender_idx}"
    elif data_name == 'dailydialog':
        # dailydialog: {'no_emotion':0, 'happiness':1, 'sadness':2, 'surprise':3, 'anger':4, 'fear':5, 'disgust':6}
        return f"说话人_{gender}"
    else:
        return f"说话人_{gender}"

def flatten_conversation_mixed_by_surrounding(conv, around_window, s_id, genders, data_name, speakers=None):
    """
    将对话按窗口大小展开，生成包含周围句子的上下文。
    如果提供了 speakers 列表，则使用其中的说话人名称；否则，通过 gender 映射获取说话人名称。
    """
    new_data = []
    for i, cur_sent in enumerate(conv):
        tmp_window = []
        for j in range(max(0, i-around_window), min(len(conv), i+around_window+1)):
            # 使用 speakers 列表中的说话人名称（中文数据）
            if speakers is not None:
                speaker_name = speakers[j]
            else:
                speaker_name = get_speaker_name(s_id, genders[j], data_name, speaker=None)
            tmp_window.append(f" {speaker_name}: {conv[j]}")
        new_data.append(tmp_window)
    return new_data

def get_label_map(data_name):
    """
    返回情感标签的中文映射。
    """
    all_data_label_map = {
        "iemocap": {0:'快乐', 1:'中性', 2:'悲伤', 3:'厌恶', 4:'愤怒', 5:'恐惧', 6:'惊讶'},
        "emorynlp":  ['快乐', '愤怒', '和平', '中性', '悲伤', '强大', '害怕'],
        "meld":      ['中性', '惊讶', '恐惧', '悲伤', '快乐', '厌恶', '愤怒'],
        "dailydialog": ['无情感', '快乐', '悲伤', '惊讶', '愤怒', '恐惧', '厌恶']
    }
    return all_data_label_map[data_name]

def preprocess_desc_speaker(str_in):
    """
    预处理说话人描述，去除特殊字符并替换换行符为空格。
    """
    str_in = str_in.split("</s>")[0].replace("<s>", "").replace("\n", " ")
    str_out = re.sub(r" {2,}", " ",  str_in)
    return str_out

def gen_default_prompting_messages(data_name, conv, around_window, s_id, desc_speaker_data=None):
    """
    生成默认的提示消息（无说话人描述）。
    """
    new_conv = []
    samples = []
    speakers = conv.get('speakers', None)  # 获取 speakers 字段，如果不存在则为 None
    for i, sent in enumerate(conv['sentences']):
        new_sent_gender = conv['genders'][i]
        # 如果有 speakers 字段，直接使用；否则通过 gender 映射
        sent_name = get_speaker_name(s_id, new_sent_gender, data_name, speaker=speakers[i] if speakers else None)
        new_sent = f'{sent_name}: {sent}'
        new_conv.append(new_sent)
    conv_str = "\n".join(new_conv)
    
    flatten_conv = flatten_conversation_mixed_by_surrounding(conv['sentences'], around_window, s_id, conv['genders'], data_name, speakers)
    
    label_map = get_label_map(data_name)
    
    for i, sent in enumerate(new_conv):
        system_msg = f'### 您是一位擅长分析对话中各说话人情感的专家。'
        conv_str = "\n".join(flatten_conv[i])
        local_context_msg = f"\n### 给定以下对话内容作为上下文：\n{conv_str}"
        speaker_name = get_speaker_name(s_id, conv["genders"][i], data_name, speaker=speakers[i] if speakers else None)
        q_msg = f'基于上述对话内容，{speaker_name} 在句子 \"{conv["sentences"][i]}\" 中的情感是什么？'
        
        label_index = conv['labels'][i]
        label_msg = label_map.get(label_index, '未知')  # 如果标签不存在，使用 '未知'

        samples.append({
            "messages":  [
                {'role': "system", 'content': system_msg + local_context_msg},
                {'role': "user", 'content': q_msg},
                {'role': "assistant", 'content': label_msg},
            ]
        })
    return samples

def gen_spdescV2_prompting_messages(data_name, conv, around_window, s_id, desc_speaker_data):
    """
    生成带有说话人描述的提示消息。
    """
    new_conv = []
    speakers = conv.get('speakers', None)  # 获取 speakers 字段，如果不存在则为 None
    for i, sent in enumerate(conv['sentences']):
        new_sent_gender = conv['genders'][i]
        # 使用 speakers 字段
        sent_name = get_speaker_name(s_id, new_sent_gender, data_name, speaker=speakers[i] if speakers else None)
        new_sent = f'{sent_name}: {sent}'
        new_conv.append(new_sent)
    conv_str = "\n".join(new_conv)
    
    flatten_conv = flatten_conversation_mixed_by_surrounding(conv['sentences'], around_window, s_id, conv['genders'], data_name, speakers)
    
    samples = []
    label_map = get_label_map(data_name)
    
    for i, sent in enumerate(new_conv):
        system_msg = f'### 您是一位擅长分析对话中各说话人情感的专家。'
        speaker_name = get_speaker_name(s_id, conv["genders"][i], data_name, speaker=speakers[i] if speakers else None)
        
        desc_str = desc_speaker_data[s_id][i].replace("\n", " ") if desc_speaker_data else ""
        desc_msg = f'\n### 给定以下关于说话人 {speaker_name} 的特征描述：\n{desc_str}' if desc_str else ""
        
        conv_str = "\n".join(flatten_conv[i])
        local_context_msg = f"\n### 给定以下对话内容作为上下文：\n{conv_str}"
        
        q_msg = f'基于上述对话内容和说话人的特征描述，{speaker_name} 在句子 \"{conv["sentences"][i]}\" 中的情感是什么？'
        
        label_index = conv['labels'][i]
        label_msg = label_map.get(label_index, '未知')  # 如果标签不存在，使用 '未知'

        samples.append({
            "messages":  [
                {'role': "system", 'content': system_msg + desc_msg + local_context_msg},
                {'role': "user", 'content': q_msg},
                {'role': "assistant", 'content': label_msg},
            ]
        })
        
    return samples 
      
def process(paths_folder_preprocessed_data, args):
    
    process_kwargs = {}
    for path_folder_preprocessed_data in paths_folder_preprocessed_data:
        
        d_type = 'train' if '.train.' in path_folder_preprocessed_data else \
                'valid' if '.valid.' in path_folder_preprocessed_data else \
                'test' if '.test.' in path_folder_preprocessed_data else None  
        
        if d_type is None:
            print(f"警告：无法识别数据类型（train/valid/test）从文件名：{path_folder_preprocessed_data}")
            continue
        
        folder_data = args.data_folder
        around_window = args.window
        data_name = args.data_name
        path_data_out = path_folder_preprocessed_data
        prompting_type = args.prompting_type
        extract_prompting_llm_id = args.extract_prompting_llm_id 
        
        raw_data = f'{folder_data}/{data_name}.{d_type}.json'
        try:
            org_data = json.load(open(raw_data, encoding='utf-8'))
        except FileNotFoundError:
            print(f"错误：原始数据文件未找到：{raw_data}")
            continue
        
        new_format = []
        
        # 如果使用说话人描述 -> 加载原始数据并预处理
        if prompting_type not in ["default"]:
            desc_speaker_data_path = f'{folder_data}/{data_name}.{d_type}_{prompting_type}_{extract_prompting_llm_id}.json'
            try:
                desc_speaker_data = json.load(open(desc_speaker_data_path, encoding='utf-8'))
            except FileNotFoundError:
                print(f"警告：描述说话人数据文件未找到：{desc_speaker_data_path}")
                desc_speaker_data = None
            processed_desc_speaker_data = {}
            if desc_speaker_data is not None and "spdesc" in prompting_type:
                for s_id, desc_all_conv in desc_speaker_data.items():
                    processed_desc_speaker_data[s_id] = [preprocess_desc_speaker(spdesc) for spdesc in desc_all_conv]
                desc_speaker_data = processed_desc_speaker_data   
        else:
            desc_speaker_data = None
            
        # 输出数据路径
        path_processed_data = raw_data.replace(".json", f".0shot_w{around_window}_{prompting_type}.jsonl") if path_data_out is None else path_folder_preprocessed_data
        
        # 提示处理函数映射
        process_function_map = {
            "spdescV2": gen_spdescV2_prompting_messages,
            "default": gen_default_prompting_messages,
        }
        
        process_func = process_function_map.get(prompting_type, process_function_map['default'])
        print(f"- process prompting by {process_func.__name__}")
        
        for s_id, conv in org_data.items(): 
            # 确保所有列表长度一致
            num_labels = len(conv['labels'])
            num_sentences = len(conv['sentences'])
            num_genders = len(conv['genders'])
            num_speakers = len(conv.get('speakers', []))
            
            # 检查是否存在 'speakers' 字段
            speakers = conv.get('speakers', None)
            if speakers is not None:
                if not (num_labels == num_sentences == num_genders == len(speakers)):
                    print(f"警告：数据条目 '{s_id}' 中的标签、句子、性别或说话人数量不一致。跳过此条目。")
                    continue
            else:
                if not (num_labels == num_sentences == num_genders):
                    print(f"警告：数据条目 '{s_id}' 中的标签、句子或性别数量不一致。跳过此条目。")
                    continue
            
            process_args = [data_name, conv, around_window, s_id, desc_speaker_data]
            try:
                samples = process_func(*process_args, **process_kwargs)
                new_format = new_format + samples
            except KeyError as e:
                print(f"错误：处理条目 '{s_id}' 时遇到 KeyError: {e}")
                continue
            except Exception as e:
                print(f"错误：处理条目 '{s_id}' 时遇到异常: {e}")
                continue
            
        with open(f'{path_processed_data}', 'wt', encoding='utf-8') as f:
            new_format = [json.dumps(e, ensure_ascii=False) for e in new_format]
            f.write("\n".join(new_format))
