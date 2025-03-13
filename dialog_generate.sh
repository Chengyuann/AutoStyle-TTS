#!/bin/bash
# txt_path='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/tts_res/ttstext.txt'
# # timbre_wav_path= '/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/timbre/test_3/roden_1_16k.wav'
# style_wav_path='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/lijiaqi/segments/denoise/denoise_jinjing_ljq_1_0h0m6dot0s_0h0m13dot0s.wav'
# timbre_wav_path='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/timbre/test_3/roden_1_16k.wav'
# style_wav_text='防晒衣还有吗？防晒衣有吗我们后面还会上给大家的，我们还会再给大家的。好不好？不用担心不用担心，谢谢大家支持。'
# result_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/result_wav/not_exp/'

# corresponding_json='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/lijiaqi/lijiaqi_jinjing_2dot06-6dot06_wav2to1.json'
corresponding_json='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/lijiaqi/lijiaqi_jinjing_6dot06-9dot15_wav2to1.json'
# corresponding_json=''
# dialogue_json='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/lijiaqi/lijiaqi2dot06-6dot6.json'
# dialogue_json='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/lijiaqi/lijiaqi6dot06-9dot15.json'
# dialogue_json='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/lijiaqi/lijiaqi2dot06-6dot6_fxl_zsl.json'
dialogue_json='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/lijiaqi/lijiaqi6dot06-9dot15_fxl_zsl.json'

style_wav_json='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/jinjing_ljq_1.json'
style_wav_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/lijiaqi/segments/10db'
# result_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/result_wav/dialogue/exp/pre4'
result_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/result_wav/dialogue/fxl_zsl/last3'



CUDA_VISIBLE_DEVICES=2 python  tts_for_dialog.py \
    --corresponding_json $corresponding_json \
    --dialogue_json $dialogue_json \
    --style_wav_json $style_wav_json \
    --style_wav_dir $style_wav_dir \
    --result_dir $result_dir 
