#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES="0,1,2,3"

# general configuration
feats_dir="../DATA/data4" #feature output dictionary
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
tag="WD-LoRA-FT"
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

config=conformer_multi_embed_decoder_asrNar.yaml
model_dir="conformer_lora_asrNar_exp7"

#init_model_dir=/ssd/zhuang/code/FunASR/examples/kespeech/conformer/exp/baseline_conformer_12e_6d_2048_256_MD
#config_file=${init_model_dir}/config.yaml
#token_list=$(yq '.tokenizer_conf.token_list' ${config_file})
#add_special_token_list=$(yq '.tokenizer_conf.add_special_token_list' ${config_file})
#init_param=${init_model_dir}/model.pt.avg10
lora_details=/ssd/zhuang/code/FunASR/examples/kespeech/conformer_lora/conf_lora/config.json
#cmvn_file=/ssd/zhuang/code/FunASR/examples/aishell/paraformer/exp/speech_paraformer_asr_nat-aishell1-pytorch/am.mvn

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
++train_data_set_list="${feats_dir}/${train_set}/audio_datasets.jsonl" \
++valid_data_set_list="${feats_dir}/${valid_set}/audio_datasets.jsonl" \
++frontend_conf.cmvn_file=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data4/WD/train/am.mvn \
++tokenizer_conf.token_list=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data4/zh_token_list/char/tokens.txt \
++tokenizer_conf.add_special_token_list=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data4/zh_token_list/char/dialects.txt \
++text_language_vocab_path=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data4/zh_token_list/char/dialects.txt \
++init_param=/ssd/zhuang/code/FunASR/examples/kespeech/conformer_lora/exp/conformer_lora_asrNar_exp4/init_model.pt \
++use_lora=true \
++lora_details="${lora_details}" \
++lora_bias=lora_only \
++output_dir="${exp_dir}/exp/${model_dir}" &> ${log_file}

#++frontend_conf.cmvn_file="${cmvn_file}" \
