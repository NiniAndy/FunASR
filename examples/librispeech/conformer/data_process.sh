#!/usr/bin/env bash

stage=0
stop_stage=3

# data
#raw_data=/data/NAS_PLUS/asr_datasets/LibriSpeech
raw_data=/data/nas/ASR_Datasets/LibriSpeech/
# wav data dir
feats_dir="../DATA/data"
config=conformer_12e_6d_2048_256.yaml

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram

set -e
set -u
set -o pipefail

train_set=train_960
workspace=`pwd`

. tools/parse_options.sh || exit 1;


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "stage 0: Data preparation"
  for x in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    # 在$feats_dir里生成wav.scp和text文件
    local/data_prep_torchaudio.sh ${raw_data}/${x} $feats_dir/${x//-/_}

    # convert wav.scp text to jsonl
    scp_file_list_arg="++scp_file_list='[\"${feats_dir}/${x//-/_}/wav.scp\",\"${feats_dir}/${x//-/_}/text\"]'"
    python ../../../funasr/datasets/audio_datasets/scp2jsonl.py \
    ++data_type_list='["source", "target"]' \
    ++jsonl_file_out=${feats_dir}/${x//-/_}/audio_datasets.jsonl \
    ${scp_file_list_arg}
  done

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then

  # 生成960h文件
  train_960=${feats_dir}/train_960
  mkdir -p "$train_960"
  train_100="${feats_dir}/train_clean_100"
  train_360="${feats_dir}/train_clean_360"
  train_500="${feats_dir}/train_other_500"

  for file_path in "$train_100"/*; do
    # 获取文件名（不包含路径）
    filename=$(basename "$file_path")

    # 定义其他两个文件夹中的对应文件路径
    file2="${train_360}/${filename}"
    file3="${train_500}/${filename}"

    # 检查其他两个文件是否存在
    if [[ -f "$file2" && -f "$file3" ]]; then
        # 拼接三个文件并保存到目标文件夹
        cat "$file_path" "$file2" "$file3" > "${train_960}/${filename}"
        echo "已拼接文件：$filename 并保存至 ${train_960}/"
    else
        echo "跳过文件：$filename，原因：在某些源文件夹中不存在该文件"
    fi
  done
  echo "文件拼接完成。"

  # 生成dev
  dev=${feats_dir}/dev
  mkdir -p "$dev"
  dev_clean="${feats_dir}/dev_clean"
  dev_other="${feats_dir}/dev_other"

  for file_path in "$dev_clean"/*; do
    # 获取文件名（不包含路径）
    filename=$(basename "$file_path")
    file2="${dev_other}/${filename}"

    # 检查其他两个文件是否存在
    if [[ -f "$file2"  ]]; then
        # 拼接三个文件并保存到目标文件夹
        cat "$file_path" "$file2"  > "${dev}/${filename}"
        echo "已拼接文件：$filename 并保存至 ${dev}/"
    else
        echo "跳过文件：$filename，原因：在某些源文件夹中不存在该文件"
    fi
  done
  echo "文件拼接完成。"

fi


dict=$feats_dir/lang_char/${train_set}/${bpemode}${nbpe}_units.txt
bpemodel=$feats_dir/lang_char/${train_set}/${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  ### Task dependent. You have to check non-linguistic symbols used in the corpus.
  echo "stage 2: Dictionary and Json Data Preparation"
  mkdir -p $feats_dir/lang_char/${train_set}

  echo "make a dictionary"
  echo "<blank> 0" > ${dict}
  echo "<s> 1" >> ${dict}
  echo "</s> 2" >> $dict

  # we borrowed these code and scripts which are related bpe from ESPnet.
  cut -f 2- -d" " $feats_dir/${train_set}/text > $feats_dir/lang_char/input.txt
  tools/spm_train --input=$feats_dir/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
  tools/spm_encode --model=${bpemodel}.model --output_format=piece < $feats_dir/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+2}' >> ${dict}
  char_num=$(wc -l < "$dict")
  echo "<unk> $char_num" >> "$dict"

  output_dict=$feats_dir/lang_char/${train_set}/${bpemode}${nbpe}.txt
  awk '{print $1}' $dict > $output_dict

fi



if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Feature and CMVN Generation"
    python ../../../funasr/bin/compute_audio_cmvn.py \
    --config-path "${workspace}/conf" \
    --config-name "${config}" \
    ++train_data_set_list="${feats_dir}/${train_set}/audio_datasets.jsonl" \
    ++cmvn_file="${feats_dir}/${train_set}/cmvn.json" \

fi
