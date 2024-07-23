#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES="0"

# general configuration
feats_dir="../DATA" #feature output dictionary
exp_dir=`pwd`
lang=zh
token_type=char
# feature configuration
nj=32

inference_device="cuda" #"cpu"
inference_checkpoint="model.pt.avg10"
inference_scp="wav.scp"
inference_batch_size=1

# exp tag
tag="Wav2Vec2-Aishell-FT"
workspace=`pwd`

master_port=12345

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev

config=wav2vec2_transformer_decoder.yaml
model_dir="baseline_$(basename "${config}" .yaml)_${tag}"
token_list=/ssd/zhuang/code/FunASR/examples/aishell/DATA/data/zh_token_list/char/tokens.txt

# ASR Training Stage
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
++output_dir="${exp_dir}/exp/${model_dir}" &> ${log_file}

