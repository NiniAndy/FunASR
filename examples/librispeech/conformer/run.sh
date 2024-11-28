#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES="0,1,2,3"

# general configuration
feats_dir="../DATA" #feature output dictionary
exp_dir=`pwd`
lang=en
token_type=bpe
stage=1
stop_stage=1

# feature configuration
nj=32

inference_device="cuda" #"cpu", "cuda:0", "cuda:1"
inference_checkpoint="model.pt.avg10"
inference_scp="wav.scp"
inference_batch_size=1


# exp tag
tag="exp1"
workspace=`pwd`

master_port=12345

#. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_960
valid_set=dev
test_sets="test_clean"

config=conformer_12e_6d_2048_256.yaml
model_dir="baseline_$(basename "${config}" .yaml)_${lang}_${token_type}_${tag}"
tokenizer=/ssd/zhuang/code/FunASR/examples/librispeech/DATA/data/lang_char/train_960/unigram5000
token_list=${tokenizer}.txt
bpe_model=${tokenizer}.model


# ASR Training Stage
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "stage 1: ASR Training"

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
  ++train_data_set_list="${feats_dir}/data/${train_set}/audio_datasets.jsonl" \
  ++valid_data_set_list="${feats_dir}/data/${valid_set}/audio_datasets.jsonl" \
  ++tokenizer_conf.token_list="${token_list}" \
  ++tokenizer_conf.bpe_model="${bpe_model}" \
  ++frontend_conf.cmvn_file="${feats_dir}/data/${train_set}/am.mvn" \
  ++output_dir="${exp_dir}/exp/${model_dir}" &> ${log_file}
fi



# Testing Stage
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 2: Inference"

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

    inference_dir="${exp_dir}/exp/${model_dir}/inference-${inference_checkpoint}/${dset}_ctc-ar-rescoring"
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
          ++tokenizer_conf.token_list="${token_list}" \
          ++tokenizer_conf.bpe_model="${bpe_model}" \
          ++frontend_conf.cmvn_file="${feats_dir}/data/${train_set}/am.mvn" \
          ++input="${_logdir}/keys.${JOB}.scp" \
          ++output_dir="${inference_dir}/${JOB}" \
          ++device="${inference_device}" \
          ++ncpu=1 \
          ++disable_log=true \
          ++batch_size="${inference_batch_size}" &> ${_logdir}/log.${JOB}.txt \
#          ++decoding_ctc_weight=1.0
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

#    echo "Computing WER ..."
#    python utils/postprocess_text_zh.py ${inference_dir}/1best_recog/text ${inference_dir}/1best_recog/text.proc
#    python utils/postprocess_text_zh.py  ${data_dir}/text ${inference_dir}/1best_recog/text.ref
#    python utils/compute_wer.py ${inference_dir}/1best_recog/text.ref ${inference_dir}/1best_recog/text.proc ${inference_dir}/1best_recog/text.cer
#    tail -n 3 ${inference_dir}/1best_recog/text.cer
  done

fi


#if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#  echo "stage 3: Computing WER ..."
#      python tools/compute-wer.py --char=1 --v=1 \
#        ${feats_dir}/data/test_clean/text /ssd/zhuang/code/FunASR/examples/librispeech/conformer/exp/baseline_conformer_12e_6d_2048_256_en_bpe_exp1/inference-model.pt.avg10/test_clean_ctc-ar-rescoring/1best_recog/text > /ssd/zhuang/code/FunASR/examples/librispeech/conformer/exp/baseline_conformer_12e_6d_2048_256_en_bpe_exp1/inference-model.pt.avg10/test_clean_ctc-ar-rescoring/1best_recog/wer
#fi