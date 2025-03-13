#!/bin/bash
txt_path='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/tts_res/ttstext.txt'
# timbre_wav_path= '/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/timbre/test_3/roden_1_16k.wav'
style_wav_path='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/lijiaqi/segments/denoise/denoise_jinjing_ljq_1_0h0m6dot0s_0h0m13dot0s.wav'
timbre_wav_path='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/timbre/test_3/roden_1_16k.wav'
style_wav_text='防晒衣还有吗？防晒衣有吗我们后面还会上给大家的，我们还会再给大家的。好不好？不用担心不用担心，谢谢大家支持。'
result_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/result_wav/not_exp/'


CUDA_VISIBLE_DEVICES=3 python  tts_with_style_and_timbre.py \
    --style_wav_path $style_wav_path \
    --style_wav_text $style_wav_text \
    --txt_path $txt_path \
    --result_dir $result_dir \
    --timbre_wav_path $timbre_wav_path \
    --is_exp True

