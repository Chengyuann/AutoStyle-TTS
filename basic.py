from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import os
import sys

os.environ['PYTHONPATH'] = 'third_party/Matcha-TTS'
sys.path.append(os.environ['PYTHONPATH'])

# cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-25Hz') # or change to pretrained_models/CosyVoice-300M for 50Hz inferencecosyvoice = CosyVoice('/apdcephfs_cq10/share_1615176/cq2/rodenluo/CosyVoice/pretrained_models/CosyVoice-300M')
cosyvoice = CosyVoice('/apdcephfs_cq10/share_1615176/cq2/rodenluo/CosyVoice/pretrained_models/CosyVoice-300M')

# zero_shot usage, <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], 22050)