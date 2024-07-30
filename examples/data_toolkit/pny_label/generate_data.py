# Author: Zhuang Xuyi
# Time: 2023/12/1 9:00
# -*- coding: utf-8 -*-
# 生成DIMNet的data_list文件
import json
import string
from tqdm import tqdm
from pypinyin import pinyin, Style, lazy_pinyin

def word2pinyin(text):
    pinyin_str = ""
    sheng_pinyin = pinyin(text, style=Style.INITIALS, strict=False)
    yun_pinyin = pinyin(text, style=Style.FINALS_TONE3)
    assert len(sheng_pinyin) == len(yun_pinyin), "声韵母长度不一致"
    for i in range(len(sheng_pinyin)):
        word2sheng = sheng_pinyin[i][0]
        word2yun = yun_pinyin[i][0]
        if word2sheng != "":
            pinyin_str += (f"{word2sheng} {word2yun} ")
        else:
            pinyin_str += (f"{word2yun} ")
    return pinyin_str

type = "test"
text_path = "/ssd/zhuang/dataset/data_KeSpeech/KeSpeech/Tasks/ASR/{}_phase1/text".format(type)
wav_scp_path = "/ssd/zhuang/dataset/data_KeSpeech/KeSpeech/Tasks/ASR/{}_phase1/wav.scp".format(type)
utt2subdialect_path = "/ssd/zhuang/dataset/data_KeSpeech/KeSpeech/Tasks/ASR/{}_phase1/utt2subdialect".format(type)
output_dir = 'data_{}_phase1.list'.format(type)

if type == "test":
    text_path = "/ssd/zhuang/dataset/data_KeSpeech/KeSpeech/Tasks/ASR/{}/text".format(type)
    wav_scp_path = "/ssd/zhuang/dataset/data_KeSpeech/KeSpeech/Tasks/ASR/{}/wav.scp".format(type)
    utt2subdialect_path = "/ssd/zhuang/dataset/data_KeSpeech/KeSpeech/Tasks/ASR/{}/utt2subdialect".format(type)
    output_dir = 'data_{}.list'.format(type)

with open(text_path, 'r', encoding='utf-8') as file:
    text_lines = file.readlines()

with open(wav_scp_path, 'r', encoding='utf-8') as file:
    wav_scp_lines = file.readlines()

with open(utt2subdialect_path, 'r', encoding='utf-8') as file:
    utt_lines = file.readlines()

id2wav = {}
for line in wav_scp_lines:
    parts = line.strip().split()
    if len(parts) == 2:
        id_, wav_path = parts
        id2wav[id_] = wav_path

id2dialect = {}
for line in utt_lines:
    parts = line.strip().split()
    if len(parts) == 2:
        id_, dialect = parts
        id2dialect[id_] = dialect

output_lines = []
for line in tqdm(text_lines):
    parts = line.strip().split()
    id = parts[0]
    text = parts[-1]
    if any(char.isdigit() for char in text) or any(char in string.punctuation for char in text):
        continue

    pinyin_str = word2pinyin(text)

    accent = id2dialect.get(id, "Unknown")
    wav = id2wav.get(id, "Unknown")
    if accent=="Unknown":
        continue
    if wav=="Unknown":
        continue

    formatted_line = json.dumps({"key": id,
                                 "wav": wav,
                                 "acc": accent,
                                 "txt": text,
                                 "pny": pinyin_str}, ensure_ascii=False)
    output_lines.append(formatted_line)


with open(output_dir, 'w', encoding='utf-8') as file:
    for line in output_lines:
        file.write(line + '\n')

print(f"Updated data saved to {output_dir}")
