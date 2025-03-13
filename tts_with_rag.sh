#!/bin/bash
corresponding_json='/apdcephfs_cq10/share_1615176/cq2/macy/tts/test_json/search_results_ab_text.json'
result_dir='/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/result_wav/rag_test/dialog_1_text/'



CUDA_VISIBLE_DEVICES=3 python  tts_with_rag.py \
    --corresponding_json $corresponding_json \
    --result_dir $result_dir 
