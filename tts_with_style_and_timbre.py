from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import os
import sys
import argparse
import random

import librosa
import soundfile as sf


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['PYTHONPATH'] = 'third_party/Matcha-TTS'
sys.path.append(os.environ['PYTHONPATH'])


def get_text(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines()]
    return lines

def tts_for_exp(args):
    '''
    实验时保存style_wav听取效果
    
    step1:tts生成style_wav
    step2:vc转换style_wav中的音色

    '''
    #load model
    cosyvoice = CosyVoice('/apdcephfs_cq10/share_1615176/cq2/rodenluo/CosyVoice/pretrained_models/CosyVoice-300M') 
    
    #load data
    style = os.path.basename(args.style_wav_path)[:-4]
    timbre = os.path.basename(args.timbre_wav_path)[:-4]
    
    lines = get_text(args.txt_path)  
    style_wav = load_wav(args.style_wav_path,16000)
    style_wav_text = args.style_wav_text
    timbre_wav = load_wav(args.timbre_wav_path,16000)
    
    # tts 
    cnt = 0
    for line in lines:
        cnt += 1
        for i, j in enumerate(cosyvoice.inference_zero_shot(line, style_wav_text, style_wav, stream=False)):
            result_wav_path = args.result_dir +'/' +  f'{style}_prompt_{cnt}.wav'
            torchaudio.save(result_wav_path.format(i), j['tts_speech'], 22050)
        # style_wav,sr = librosa.load(result_wav_path,sr = 22050)
        # style_wav_resample = librosa.resample(style_wav,orig_sr = sr,target_sr = 16000)
        style_wav_22050 = load_wav(result_wav_path,22050)
        style_wav_resample = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000)(style_wav_22050)
        
        print('timbre_wav:',timbre_wav.shape)
        print('style_wav_resample:',style_wav_resample.shape)
        for i, j in enumerate(cosyvoice.inference_vc(style_wav_resample, timbre_wav, stream=False)):
            result_wav_path = args.result_dir + f'{style}_{cnt}_to_{timbre}_exp.wav'
            torchaudio.save(result_wav_path, j['tts_speech'], 22050)
    
    
    
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
    style = os.path.basename(args.style_wav_path)[:-4]
    # timbre = os.path.basename(args.timbre_wav_path)[:-4]
    timbre = os.path.basename(args.timbre_wav_path)[:-4]
    
    
    lines = get_text(args.txt_path)  
    style_wav = load_wav(args.style_wav_path,16000)
    style_wav_text = args.style_wav_text
    # timbre_wav = load_wav(args.timbre_wav_path,16000)
    timbre_wav = load_wav(args.timbre_wav_path,16000)
    
    
    # tts 
    cnt = 0
    for line in lines:
        cnt += 1
        for i, j in enumerate(cosyvoice.inference_tts_with_st(line, style_wav_text, style_wav, timbre_wav,stream=False)):
            result_wav_path = args.result_dir +'/' +  f'{style}_{cnt}_to_{timbre}.wav'
            torchaudio.save(result_wav_path.format(i), j['tts_speech'], 22050)
    
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
    parser.add_argument('--style_wav_path', required=True, help='style wav (exp:pdd or jj_ljq)')
    parser.add_argument('--timbre_wav_path', required=True, help='timbre wav (exp:test80)')
    
    parser.add_argument('--style_wav_text', required=True, help='style text ')
    
    parser.add_argument('--txt_path', required=True, help='text for tts')
    parser.add_argument('--result_dir', required=True, help='path to save results')
    
    parser.add_argument('--is_exp', type=bool,default = False,help='path to save results')    
    
    
    

    args = parser.parse_args()
    main(args)