from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import os
import sys
import argparse
import random
import json
import subprocess
import re
import torch

import librosa
import soundfile as sf
from datetime import datetime
from tqdm import tqdm

os.environ['PYTHONPATH'] = 'third_party/Matcha-TTS'
sys.path.append(os.environ['PYTHONPATH'])

class JsonDataReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()

    def load_data(self):
        """从文件中加载 JSON 数据并返回列表"""
        with open(self.file_path, 'r', encoding='utf-8') as file:
            data = []
            for line in file:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        # print(data[0])            
        return data

    def get_zh_text_by_index(self, index):
        """根据索引获取 zh_text"""
        if 0 <= index <= len(self.data):
            return self.data[index-1]['zh_text']
        else:
            return "索引超出范围"
        
    def get_timbre_wav(self,speaker):
        if speaker == 'jinjing':
            # timbre_wav = load_wav('/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/lijiaqi/segments/denoise/denoise_jinjing_ljq_1_0h7m41dot0s_0h7m44dot0s.wav',16000)
            timbre_wav = load_wav('/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/timbre/test_3/zsl/aimeelzhang_1_16k.wav',16000)
        elif speaker == 'lijiaqi':
            # timbre_wav = load_wav('/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/lijiaqi/segments/denoise/denoise_jinjing_ljq_1_0h0m6dot0s_0h0m13dot0s.wav',16000)
            # timbre_wav = load_wav('/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/timbre/test80/16k/符晓乐003_240_16k.wav',16000)
            timbre_wav = load_wav('/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/timbre/test_3/16k/elricfu_1_16k.wav',16000)
        return timbre_wav

    def get_fileid(self,value):
        return self.data[value-1]['file_id']
        
        

def get_text(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines()]
    return lines



def get_timbre_wav_path(speaker):
    if speaker == 'w1':
            timbre_wav_path = '/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/rag_test/youtube4/timbre/engagement_0h0m4dot0s_0h0m9dot51s_w1.wav'
    elif speaker == 'w2':
            timbre_wav_path = '/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/rag_test/youtube4/timbre/engagement_0h0m10dot51s_0h0m13dot0s_w2.wav'
    elif speaker == 'm1':
            timbre_wav_path = '/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/rag_test/youtube4/timbre/engagement_0h0m16dot73s_0h0m21dot34s_m1.wav'
    elif speaker == 'm2':
            timbre_wav_path = '/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/rag_test/youtube4/timbre/engagement_0h0m13dot18s_0h0m16dot73s_m2.wav'
    return timbre_wav_path
    
def get_text_and_wav(corr_json_path):
    '''
    目标：
    根据json对应 获得相应style_wav,style_wav_text,timbre_wav,text 组成列表并返回
    '''
    data_list = []
    with open(corr_json_path, 'r', encoding='utf-8') as file:
        # data = json.load(file)
        for line in file:
            data = json.loads(line)
            tts_text = data.get('zh_text')
            speaker = data.get('speaker')
            timbre_wav_path = get_timbre_wav_path(speaker)
            file_id = data.get('retrieved_file_id')
            style_wav_path = data.get('retrieved_file_id')
            style_wav_text = data.get('retrieved_text')
            is_whisper = data.get('whisper')
            item = {'is_whisper':is_whisper,'tts_text': tts_text, 'speaker':speaker,'timbre_wav_path': timbre_wav_path,'style_wav_path':style_wav_path,'style_wav_text':style_wav_text}
            data_list.append(item)     
    return  data_list

def tts_for_exp(args):
    '''
    实验时保存style_wav听取效果
    
    step1:tts生成style_wav
    step2:vc转换style_wav中的音色

    '''
    #load model
    cosyvoice = CosyVoice('/apdcephfs_cq10/share_1615176/cq2/rodenluo/CosyVoice/pretrained_models/CosyVoice-300M') 
    
    #load data
    correspond_json_path = args.corresponding_json
    dialogue_json_path = args.dialogue_json
    style_wav_json_path = args.style_wav_json
    style_wav_dir = args.style_wav_dir
    
    dialogue_data = JsonDataReader(dialogue_json_path)
    style_wav_json_data = JsonDataReader(style_wav_json_path)
    
    with open(correspond_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    cnt = 0
    for key, value in tqdm(data.items()):
        # print(f"Key: {key}, Value: {value['value']}, Speaker: {value['speaker']}, Emotion: {value['emotion']}")
        if value == 'null':
            continue
        zh_text = dialogue_data.get_zh_text_by_index(int(key))
        timbre_wav = dialogue_data.get_timbre_wav(value['speaker'])
        style_wav_fileid = style_wav_json_data.get_fileid(int(value['value']))
        style_wav_text = style_wav_json_data.get_zh_text_by_index(int(value['value']))
        style_wav = load_wav(style_wav_dir + '/' + style_wav_fileid + '.wav',16000)
        
        print(zh_text,style_wav_text,value['speaker'])
        for i, j in enumerate(cosyvoice.inference_zero_shot(zh_text, style_wav_text, style_wav, stream=False)):
            result_wav_path = args.result_dir +'/' +  f'{style_wav_fileid}_prompt_{cnt}'
            torchaudio.save(result_wav_path+'_{}'.format(i)+'.wav', j['tts_speech'], 22050)
        style_wav_22050 = load_wav(result_wav_path,22050)
        style_wav_resample = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000)(style_wav_22050)
        
        print('timbre_wav:',timbre_wav.shape)
        print('style_wav_resample:',style_wav_resample.shape)
        for i, j in enumerate(cosyvoice.inference_vc(style_wav_resample, timbre_wav, stream=False)):
            timbre = value['speaker']
            result_wav_path = args.result_dir +'/'+ f'{style_wav_fileid}_{cnt}_to_{timbre}_exp'
            torchaudio.save(result_wav_path + '_{}'.format(i)+'.wav', j['tts_speech'], 22050)
    
    
    
    return 

def tts_for_infer(args):
    '''
    实际推理直接style_token+timbre_token ---> result_wav
    
    step1:tts生成style token
    step2:style token + timbre token (embed + timbre mel) ---> result_wav

    '''
   #load model
    cosyvoice = CosyVoice('/apdcephfs_cq10/share_1615176/cq2/rodenluo/CosyVoice/pretrained_models/CosyVoice-300M') 
    
    #load data
    correspond_json_path = args.corresponding_json
    result_dir = args.result_dir
        
    current_time = datetime.now().strftime("%m%d%H%M")
    result_dir = result_dir + f'_{current_time}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    data_list = get_text_and_wav(correspond_json_path)
    cnt = 0
    for item in data_list:
        print(item)
        cnt += 1
        # # print(f"Key: {key}, Value: {value['value']}, Speaker: {value['speaker']}, Emotion: {value['emotion']}")
        tts_text = item['tts_text']
        style_wav_text = item['style_wav_text']
        is_whisper = item['is_whisper']
        if is_whisper:
            style_wav= load_wav(item['style_wav_path'],16000)
            timbre_wav= load_wav('/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/rag_test/youtube4/timbre/engagement_0h0m46dot41s_0h0m48dot99s.wav',16000) 
            # style_wav_text = 'I think i made a mistake.'
            # style_wav = torch.cat((style_wav_1,style_wav_2),dim=1)      
        else:
            style_wav = load_wav(item['style_wav_path'],16000)
            timbre_wav = load_wav(item['timbre_wav_path'],16000)
            
        # timbre_wav = dialogue_data.get_timbre_wav(value['speaker'])
        style_wav_fileid = os.path.basename(item['style_wav_path'])[:-4]
        # style_wav_text = style_wav_json_data.get_zh_text_by_index(int(value['value']))
        # style_wav = load_wav(style_wav_dir + '/' + style_wav_fileid + '.wav',16000)
        timbre = item['speaker']
        # print(zh_text,style_wav_text,value['speaker'])
        
        for i, j in enumerate(cosyvoice.inference_tts_with_st(tts_text, style_wav_text, style_wav, timbre_wav,stream=False)):
            result_wav_path = result_dir +'/' +  f'{cnt}_{style_wav_fileid}_to_{timbre}'
            torchaudio.save(result_wav_path+'_{}.wav'.format(i), j['tts_speech'], 22050)
            
        
    # tts 
    # cnt = 0
    # for line in lines:
    #     cnt += 1
    #     for i, j in enumerate(cosyvoice.inference_tts_with_st(line, style_wav_text, style_wav, timbre_wav,stream=False)):
    #         result_wav_path = args.result_dir +'/' +  f'{style}_{cnt}_to_{timbre}.wav'
    #         torchaudio.save(result_wav_path.format(i), j['tts_speech'], 22050)
    
    return

def main(args):
    print('flag:',args.is_exp)
    
    if args.is_exp:
        print('---exp---')
        tts_for_exp(args)
    else:
        
        print('---not exp---')
        tts_for_infer(args)
    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate vc result from style_dir to timbre_dir.")
    ## corr:text,speaker(timbre),style
    parser.add_argument('--corresponding_json', required=True, help='include text style timbre information')
    # parser.add_argument('--dialogue_json', required=True, help='provide text')
    parser.add_argument('--result_dir', required=True, help='path to save results')
    
    parser.add_argument('--is_exp', type=bool,default = False,help='path to save results')    
    
    

    args = parser.parse_args()
    main(args)