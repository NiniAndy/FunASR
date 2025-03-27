ref=/ssd/zhuang/code/SLAM-LLM/examples/asr_librispeech/exp/vicuna-7b-v1.5-librispeech-linear-steplrwarmupkeep1e-4-whisper-medium-20250315/asr_epoch_2_step_67423/decode_test_clean_beam4_gt
test_dir=/ssd/zhuang/code/SLAM-LLM/examples/asr_librispeech/exp/vicuna-7b-v1.5-librispeech-linear-steplrwarmupkeep1e-4-whisper-medium-20250315/asr_epoch_2_step_67423/decode_test_clean_beam4_pred
wer_dir=/ssd/zhuang/code/SLAM-LLM/examples/asr_librispeech/exp/vicuna-7b-v1.5-librispeech-linear-steplrwarmupkeep1e-4-whisper-medium-20250315/asr_epoch_2_step_67423/decode_test_clean_beam4_wer


python tools/compute-wer-wenet-version.py --char=1 --v=1 \
  $ref $test_dir > $wer_dir