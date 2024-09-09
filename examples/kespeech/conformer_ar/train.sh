#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES="0,1,2,3"

# general configuration
feats_dir="../DATA/data5" #feature output dictionary
exp_dir=`pwd`

stage=1
stop_stage=1

# feature configuration
nj=32

inference_device="cuda" #"cpu", "cuda:0", "cuda:1"
inference_checkpoint="model.pt.avg10"
inference_scp="wav.scp"
inference_batch_size=1

# exp tag
tag="ar_only_2"
workspace=`pwd`

master_port=12345

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


train_set=WD/train
valid_set=WD/dev
# test_sets=WD/test
#test_sets="ES/Beijing/test ES/Ji-Lu/test ES/Jiang-Huai/test ES/Jiao-Liao/test ES/Lan-Yin/test ES/Northeastern/test ES/Southwestern/test ES/Zhongyuan/test MD/test"


config=conformer_ar_only.yaml
model_dir="baseline_$(basename "${config}" .yaml)_${tag}"


# ASR Training Stage
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "stage 4: ASR Training"

  mkdir -p ${exp_dir}/exp/${model_dir}
  current_time=$(date "+%Y-%m-%d_%H-%M")
  log_file="${exp_dir}/exp/${model_dir}/train.log.txt.${current_time}"
  echo "log_file: ${log_file}"


  export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
  gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  torchrun \
  --nnodes 1 \
  --nproc_per_node ${gpu_num} \
  --master_port ${master_port} \
  ../../../funasr/bin/train.py \
  --config-path "${workspace}/conf" \
  --config-name "${config}" \
  ++train_data_set_list=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data5/WD/train/audio_datasets.jsonl \
  ++valid_data_set_list=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data5/WD/dev/audio_datasets.jsonl \
  ++tokenizer_conf.token_list=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data5/zh_token_list/char/tokens.txt \
  ++text_language_vocab_path=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data5/zh_token_list/char/dialects.txt \
  ++frontend_conf.cmvn_file=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data5/WD/train/am.mvn \
  ++init_param=/ssd/zhuang/code/FunASR/examples/kespeech/conformer_ar/exp/baseline_conformer_ar_only_ar_only/init_conformer_ar_only.pt \
  ++output_dir="${exp_dir}/exp/${model_dir}" &> ${log_file}


#  ++frontend_conf.anchor_file=/ssd/zhuang/code/FunASR/examples/aishell2/conformer/exp/conformer_aishell2/am.mvn \
#  ++tokenizer_conf.add_special_token_list=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data5/zh_token_list/char/dialects.txt \
fi



# Testing Stage
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 5: Inference"

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

    inference_dir="${exp_dir}/exp/${model_dir}/inference-${inference_checkpoint}/${dset}_funasr_ar_decoder"
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
          --config-path="${exp_dir}/exp/${model_dir}" \
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

    echo "Computing WER ..."
    python utils/postprocess_text_zh.py ${inference_dir}/1best_recog/text ${inference_dir}/1best_recog/text.proc
    python utils/postprocess_text_zh.py  ${data_dir}/text ${inference_dir}/1best_recog/text.ref
    python utils/compute_wer.py ${inference_dir}/1best_recog/text.ref ${inference_dir}/1best_recog/text.proc ${inference_dir}/1best_recog/text.cer
    tail -n 3 ${inference_dir}/1best_recog/text.cer
  done

fi
