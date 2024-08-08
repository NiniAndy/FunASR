#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES="0,1,2,3"
inference_device="cuda" #"cpu", "cuda:0", "cuda:1"
save_root="/ssd/zhuang/code/FunASR/examples/kespeech/conformer_ar/spurious_label"
feats_dir="/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data3"
inference_scp="wav.scp"

config_path="/ssd/zhuang/code/FunASR/examples/aishell2/conformer/exp/conformer_aishell2"
cmvn_file="/ssd/zhuang/code/FunASR/examples/aishell2/conformer/exp/conformer_aishell2/am.mvn"


. utils/parse_options.sh || exit 1;

test_sets="ES/Beijing/train ES/Ji-Lu/train ES/Jiang-Huai/train ES/Jiao-Liao/train ES/Lan-Yin/train ES/Northeastern/train ES/Southwestern/train ES/Zhongyuan/train MD/train"


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

  inference_dir="${save_root}/${dset}"
  _logdir="${inference_dir}/logdir"
  echo "inference_dir: ${inference_dir}"

  mkdir -p "${_logdir}"
  data_dir="${feats_dir}/${dset}"
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
        --config-path="${config_path}" \
        --config-name="config.yaml" \
        ++init_param="${exp_dir}/exp/${model_dir}/${inference_checkpoint}" \
        ++tokenizer_conf.token_list="${token_list}" \
        ++frontend_conf.cmvn_file="${feats_dir}/${train_set}/am.mvn" \
        ++input="${_logdir}/keys.${JOB}.scp" \
        ++output_dir="${inference_dir}/${JOB}" \
        ++device="${inference_device}" \
        ++ncpu=1 \
        ++disable_log=true \
        ++batch_size="${inference_batch_size}" &> ${_logdir}/log.${JOB}.txt
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
done


