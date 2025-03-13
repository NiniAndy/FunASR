ref=/ssd/zhuang/code/FunASR/examples/librispeech/DATA/data/test_clean/text
test_dir=/ssd/zhuang/code/FunASR/examples/industrial_data_pretraining/llm_asr/exp/llm_asr_whisper_linear_qwen_ls960_exp3/inference-model.pt.ep4/test_clean/1best_recog



python tools/compute-wer-wenet-version.py --char=1 --v=1 \
  $ref $test_dir/text > $test_dir/wer