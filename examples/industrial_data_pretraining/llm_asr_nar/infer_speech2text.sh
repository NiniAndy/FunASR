
# Authority sh
#python funasr/bin/inference.py \
#--config-path="/nfs/zhifu.gzf/ckpt/llm_asr_nar_exp1" \
#--config-name="config.yaml" \
#++init_param="/nfs/zhifu.gzf/ckpt/llm_asr_nar_exp1/model.pt.ep5" \
#++input="/Users/zhifu/funasr1.0/test_local/data_tmp/tmp_wav_10.jsonl" \
#++output_dir="/nfs/zhifu.gzf/ckpt/llm_asr_nar_exp1/inference/aishell2-dev_ios-funasr" \
#++device="cpu"

# solo demo
#test_data=/ssd/zhuang/code/FunASR/examples/librispeech/DATA/data/test_clean/audio_datasets.jsonl
#root=/ssd/zhuang/code/FunASR/examples/industrial_data_pretraining/llm_asr/exp/llm_asr_whisper_linear_qwen_ls960_exp
#log_file=${root}/test.log.txt
#echo "log_file: ${log_file}"
#
#python ../../../funasr/bin/inference.py \
#--config-path="${root}" \
#--config-name="config.yaml" \
#++init_param="${root}/model.pt.best" \
#++input="${test_data}" \
#++output_dir="${root}" \
#++device="cuda" &> ${log_file}


# multi demo
inference_device="cuda" #"cpu", "cuda:0", "cuda:1"
CUDA_VISIBLE_DEVICES="0,1,2,3"
# dataset
feats_dir="/ssd/zhuang/code/FunASR/examples/librispeech/DATA" #feature output dictionary
#test_sets="dev_clean dev_other test_clean test_other"
test_sets="test_clean test_other"
#test_sets="test"
inference_scp="wav.scp"
# model detail
exp_dir=/ssd/zhuang/code/FunASR/examples/industrial_data_pretraining/llm_asr
model_dir=llm_asr_whisper_linear_qwen_ls960_exp3
inference_checkpoint=model.pt.best

if [ ${inference_device} == "cuda" ]; then
    nj=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
else
    inference_batch_size=1
    CUDA_VISIBLE_DEVICES=""
    for JOB in $(seq ${nj}); do
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"-1,"
    done
fi


for dset in ${test_sets}; do
  inference_dir="${exp_dir}/exp/${model_dir}/inference-${inference_checkpoint}/${dset}"
  _logdir="${inference_dir}/logdir"
  echo "inference_dir: ${inference_dir}"

  mkdir -p "${_logdir}"
  data_dir="${feats_dir}/data/${dset}"
  key_file=${data_dir}/${inference_scp}

  split_scps=
  for JOB in $(seq "${nj}"); do
      split_scps+=" ${_logdir}/keys.${JOB}.scp"
  done
  utils/split_scp.pl "${key_file}" ${split_scps}

  gpuid_list_array=(${CUDA_VISIBLE_DEVICES//,/ })
  for JOB in $(seq ${nj}); do
      {
        id=$((JOB-1))
        gpuid=${gpuid_list_array[$id]}
        export CUDA_VISIBLE_DEVICES=${gpuid}
        python ../../../funasr/bin/inference.py \
        --config-path="${exp_dir}/exp/${model_dir}" \
        --config-name="config.yaml" \
        ++init_param="${exp_dir}/exp/${model_dir}/${inference_checkpoint}" \
        ++input="${_logdir}/keys.${JOB}.scp" \
        ++output_dir="${inference_dir}/${JOB}" \
        ++device="${inference_device}" &> ${_logdir}/log.${JOB}.txts
      }&

  done
  wait

  mkdir -p ${inference_dir}/1best_recog
  for f in token score text; do
      if [ -f "${inference_dir}/${JOB}/1best_recog/${f}" ]; then
        for JOB in $(seq "${nj}"); do
            cat "${inference_dir}/${JOB}/1best_recog/${f}"
        done | sort -k1 >"${inference_dir}/1best_recog/${f}"
      fi
  done

#  # CN WER
#  echo "Computing WER ..."
#  python utils/postprocess_text_zh.py ${inference_dir}/1best_recog/text ${inference_dir}/1best_recog/text.proc
#  python utils/postprocess_text_zh.py  ${data_dir}/text ${inference_dir}/1best_recog/text.ref
#  python utils/compute_wer.py ${inference_dir}/1best_recog/text.ref ${inference_dir}/1best_recog/text.proc ${inference_dir}/1best_recog/text.cer
#  tail -n 3 ${inference_dir}/1best_recog/text.cer
done