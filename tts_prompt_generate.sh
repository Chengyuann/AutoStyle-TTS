txt_path='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/tts_res/ttstext.txt'
prompt_wav_path='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/lijiaqi/segments/denoise/denoise_jinjing_ljq_1_0h0m6dot0s_0h0m13dot0s.wav'
prompt_wav_text='防晒衣还有吗？防晒衣有吗我们后面还会上给大家的，我们还会再给大家的。好不好？不用担心不用担心，谢谢大家支持。'
result_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/tts_res/ljq'


CUDA_VISIBLE_DEVICES=3 python  tts_from_lines.py --txt_path $txt_path \
    --prompt_wav_path $prompt_wav_path \
    --prompt_wav_text $prompt_wav_text \
    --result_dir $result_dir \