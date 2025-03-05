
# Authority sh
#python funasr/bin/inference.py \
#--config-path="/nfs/zhifu.gzf/ckpt/llm_asr_nar_exp1" \
#--config-name="config.yaml" \
#++init_param="/nfs/zhifu.gzf/ckpt/llm_asr_nar_exp1/model.pt.ep5" \
#++input="/Users/zhifu/funasr1.0/test_local/data_tmp/tmp_wav_10.jsonl" \
#++output_dir="/nfs/zhifu.gzf/ckpt/llm_asr_nar_exp1/inference/aishell2-dev_ios-funasr" \
#++device="cpu"

test_data=/ssd/zhuang/code/FunASR/examples/aishell/DATA/data/test/audio_datasets.jsonl
root=/ssd/zhuang/code/FunASR/examples/industrial_data_pretraining/llm_asr/exp/llm_asr_whisper_qwen_aishell1_exp
log_file=${root}/test.log.txt
echo "log_file: ${log_file}"

python ../../../funasr/bin/inference.py \
--config-path="${root}" \
--config-name="config.yaml" \
++init_param="${root}/model.pt.best" \
++input="${test_data}" \
++output_dir="${root}" \
++device="cuda" &> ${log_file}