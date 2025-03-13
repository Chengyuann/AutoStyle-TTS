#!/bin/bash

#generate tts_res
# python /apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/CosyVoice/basic_usage.py 

# style_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/pdd/segments'
# style_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/tts_res/ljq'
# style_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/tbh/segments/neutral/'
# style_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/seedtts_testset/zh/tmp_style'
style_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/lijiaqi/segments/16kHz/'
# style_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/YuanMengZhiXing/16k/'
# timbre_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/timbre/test_3/16k/'
# timbre_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/timbre/test_3/zsl/'
# timbre_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/timbre/test80/temp/'
# timbre_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/seedtts_testset/zh/tmp_timbre'
# timbre_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/YuanMengZhiXing/16k/'
timbre_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/seedtts_testset/zh/prompt_temp'
# timbre_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/tbh/segments/neutral/'
# timbre_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/timbre/test80/tongshi/'
# result_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/result_wav/pdd_to_test80'
# result_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/result_wav/single_sentence/sby_to_tongshi/'
result_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/seedtts_testset/zh/cosy_res/ljq_to_seed'
style_num=10
timbre_num=10
# txt_path='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/YuanMengZhiXing/tts.txt'
txt_path='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/seedtts_testset/zh/tts_seed.txt'


CUDA_VISIBLE_DEVICES=1 python  /apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/CosyVoice/vc_from_dir_seed.py --style_dir $style_dir \
    --timbre_dir $timbre_dir \
    --result_dir $result_dir \
    --style_num $style_num \
    --txt_path $txt_path \
    --timbre_num $timbre_num 