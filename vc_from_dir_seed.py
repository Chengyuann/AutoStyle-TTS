from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import os
import sys
import argparse
import random
import json


# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['PYTHONPATH'] = 'third_party/Matcha-TTS'
sys.path.append(os.environ['PYTHONPATH'])

# cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT')
# # sft usage
# print(cosyvoice.list_avaliable_spks())
# # change stream=True for chunk stream inference
# for i, j in enumerate(cosyvoice.inference_sft('你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？', '中文女', stream=False)):
#     torchaudio.save('sft_{}.wav'.format(i), j['tts_speech'], 22050)

# prompt_speech_16k = load_wav('/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/CosyVoice/zero_shot_prompt.wav',16000)
# source_speech_16k = load_wav('/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_wav/jj_prompt1_hwk.wav', 16000)
# for i, j in enumerate(cosyvoice.inference_vc(source_speech_16k, prompt_speech_16k, stream=False)):
#     torchaudio.save('/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/result_wav/jj_to_zero_hwk_1.wav'.format(i), j['tts_speech'], 22050)
    
def get_path(dir,num):
    if not os.path.isdir(dir):
        raise ValueError(f"'{dir}' 不是一个有效的目录")
    all_files = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    if num > len(all_files):
        raise ValueError(f"请求的文件数量 {num} 超过可用文件数量 {len(all_files)}")
    selected_files = random.sample(all_files, num)

    return selected_files    


def get_text(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines()]
    return lines

def get_style_wav_text(json_path,file_id):
        """从文件中加载 JSON 数据并返回列表"""
        print(file_id)
        file_id = 'denoise_' + file_id
        with open(json_path, 'r', encoding='utf-8') as file:
            # data=file.readlines()
            data = json.load(file)
            # print(data)
            for entry in data:
                if entry['file_id'] ==file_id:
                    zh_text = entry['zh_text']
                    break
                            
        return zh_text
def get_style_and_text(lst_path,num):
    # 读取 .lst 文件中的数据
    data = []

    with open(lst_path, 'r') as file:
        for line in file:
            # 去掉行末的换行符并按 '|' 分割
            parts = line.strip().split('|')
            if len(parts) >= 4:
                # 提取 c 和 d 列（假设 c 是第 3 列，d 是第 4 列）
                b = parts[1]
                c = parts[2]
                data.append((c, b))

    # 随机选择 100 条数据
    if len(data) >= num:
        sampled_data = random.sample(data, num)
    else:
        print(f"数据不足 100 条，无法随机提取 {num} 条数据。")
        sampled_data = data
    return sampled_data
    
def main(args):
    style_dir = args.style_dir
    timbre_dir = args.timbre_dir
    result_dir = args.result_dir
    style_num = args.style_num
    timbre_num = args.timbre_num
    
    # ### get style and timbre wav in random
    # print(style_num)
    # # style_wav_paths = get_path(style_dir,style_num)
    # ### version_1
    # lst_path = '/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/seedtts_testset/zh/meta.lst'
    # style_and_text = get_style_and_text(lst_path,50)
    
    # ### version_2
    # timbre_wav_paths = get_path(timbre_dir,timbre_num)
    # print(len(style_and_text),len(timbre_wav_paths))
    
    # ### generate
    cosyvoice = CosyVoice('/apdcephfs_cq10/share_1615176/cq2/rodenluo/CosyVoice/pretrained_models/CosyVoice-300M')
    lines = get_text(args.txt_path)
    
    # data =[]
    # ### version_1:
    # for style_wav_path,style_wav_text in style_and_text:
    #     style_wav_path = '/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/seedtts_testset/zh/' + style_wav_path.replace('prompt-wavs','prompt-wavs-16k')
    #     print(style_wav_path)
    #     print(style_wav.shape)
    #     style_wav = load_wav(style_wav_path, 16000)
    #     style =  os.path.basename(style_wav_path)[:-4]
    #     # style_json_path = '/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/tbh/tbh_neutral.json'
    #     # style_json_path = '/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/seedtts_testset/zh/tmp_style.json'
    #     # print(style_wav.shape)
    #     # style_json_path = style_wav_path.replace('16k','json').replace('wav','json').replace('_json','')
    #     # style_wav_text = get_style_wav_text(style_json_path,style)
    #     for timbre_wav_path in timbre_wav_paths:
    #         timbre_wav = load_wav(timbre_wav_path, 16000)
    #         cnt = 0
    #         for line in lines:
    #             cnt += 1
    #             print(line,'+',style_wav_text)
    #             for i, j in enumerate(cosyvoice.inference_zero_shot(line,style_wav_text,style_wav, stream=False)):
    #                 timbre = os.path.basename(timbre_wav_path)[:-4]
                    
    #                 # result_wav_path = result_dir + f'/{style}_to_{timbre}_{cnt}'
    #                 result_wav_path = result_dir + f'/{style}_to_{cnt}'
    #                 torchaudio.save(result_wav_path+'_new_{}.wav'.format(i), j['tts_speech'], 22050)
    #             # for cal_sim meta.lst
    #             a = os.path.basename(result_wav_path+'_new_{}.wav')[:-4]
    #             b = style_wav_text
    #             c = style_wav_path
    #             d = line
    #             data.append([a,b,c,d])
            
    # filename = f'{result_dir}/meta.lst'

    # # 打开文件进行写入
    # with open(filename, 'w') as file:
    #     for row in data:
    #         # 将列表中的元素用 '|' 连接成字符串
    #         line = '|'.join(row)
    #         # 写入文件并换行
    #         file.write(line + '\n')
    # print(f"数据已写入 {filename}")
    # return None


    ### version_1:
    # data = []
    # # style_json = '/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/jinjing_ljq_1.json'
    # meta_lst = '/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/seedtts_testset/zh/meta.lst'
    # style_and_text = get_style_and_text(meta_lst,100)
    
    # for item in style_and_text:
    #     style_wav_path = '/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/seedtts_testset/zh/' + item[0].replace('-wavs','_temp').replace('.wav','_16k.wav')
    #     style_wav_text = item[1]
    #     style = os.path.basename(item[0])[:-4]
    #     style_wav = load_wav(style_wav_path,16000)
    #     cnt = 0
    #     for line in lines:
    #         cnt += 1
    #         for i, j in enumerate(cosyvoice.inference_zero_shot(line,style_wav_text,style_wav, stream=False)):
    #                 # result_wav_path = result_dir + f'/{style}_to_{timbre}_{cnt}'
    #                 result_wav_path = result_dir + f'/{style}_to_{cnt}'
    #                 torchaudio.save(result_wav_path+'_new_{}.wav'.format(i), j['tts_speech'], 22050)
    #         # for cal_sim meta.lst
    #         a = os.path.basename(result_wav_path+'_new_{}.wav')[:-4]
    #         b = style_wav_text
    #         c = style_wav_path
    #         d = line
    #         data.append([a,b,c,d])
    # # 打开文件进行写入
    # filename = f'{result_dir}/meta.lst'
    # with open(filename, 'w',encoding='utf-8') as file:
    #     for row in data:
    #         # 将列表中的元素用 '|' 连接成字符串
    #         line = '|'.join(row)
    #         # 写入文件并换行
    #         file.write(line + '\n')
    # print(f"数据已写入 {filename}")
    
    ### version 2
    data2 = []
    meta_lst = '/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/seedtts_testset/zh/meta.lst'
    style_wav_paths = get_path(style_dir,style_num)
    timbre_wav_paths = get_style_and_text(meta_lst,timbre_num)
    
    for style_wav_path in style_wav_paths:
        style_wav = load_wav(style_wav_path,16000)
        style_json = '/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/jinjing_ljq_1.json'
        style =  os.path.basename(style_wav_path)[:-4]
        style_wav_text = get_style_wav_text(style_json,style)
        
        for timbre_wav_part in timbre_wav_paths:
            timbre_wav_path = '/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/seedtts_testset/zh/' + timbre_wav_part[0].replace('-wavs','_temp').replace('.wav','_16k.wav')
            timbre_wav = load_wav(timbre_wav_path,16000)
            timbre = os.path.basename(timbre_wav_path)[:-4]
            cnt = 0
            for line in lines:
                cnt += 1
                for i, j in enumerate(cosyvoice.inference_tts_with_st(line,style_wav_text,style_wav,timbre_wav,stream=False)):
                    # result_wav_path = result_dir + f'/{style}_to_{timbre}_{cnt}'
                    result_wav_path = result_dir + f'/{style}_to_{timbre}_{cnt}'
                    torchaudio.save(result_wav_path+'_new.wav'.format(i), j['tts_speech'], 22050)
                
                        # for cal_sim meta.lst
                a = os.path.basename(result_wav_path+'_new.wav')[:-4]
                b = style_wav_text
                c = timbre_wav_path
                d = line
                data2.append([a,b,c,d])
    # 打开文件进行写入
    filename = f'{result_dir}/meta.lst'
    with open(filename, 'w',encoding='utf-8') as file:
        for row in data2:
            # 将列表中的元素用 '|' 连接成字符串
            line = '|'.join(row)
            # 写入文件并换行
            file.write(line + '\n')
    print(f"数据已写入 {filename}")
    
    
    return None

    
    
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate vc result from style_dir to timbre_dir.")
    parser.add_argument('--txt_path', required=True, help='tts text')
    parser.add_argument('--style_dir', required=True, help='style dir (exp:pdd or jj_ljq)')
    parser.add_argument('--timbre_dir', required=True, help='timbre dir (exp:test80)')
    parser.add_argument('--result_dir', required=True, help='path to save results')
    
    parser.add_argument('--style_num', type=int,required=True, help='path to save results')
    parser.add_argument('--timbre_num', type=int,required=True, help='path to save results')
    
    
    
    

    args = parser.parse_args()
    main(args)