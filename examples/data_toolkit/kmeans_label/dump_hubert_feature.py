import os

import fairseq
import numpy as np
import torch
import tqdm
import yaml
import json
from npy_append_array import NpyAppendArray

from mfcc_extractor import load_audio

# 读取配置文件
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


class Reader(object):
    def __init__(self, ckpt_path, layer, max_chunk=1600000):
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().cuda()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk

    def get_feats(self, path):
        waveform, _ = load_audio(path)
        feat = []
        feat_chunk, _ = self.model.extract_features(source=waveform.cuda(),
                                                    padding_mask=None,
                                                    mask=False,
                                                    output_layer=self.layer)
        feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)



def get_path_iterator(lines):
        def iterate():
            for item in lines:
                item = item.strip()
                item = json.loads(item)
                yield item
        return iterate, len(lines)



##################  main  #####################
# 传入需要被提取的文件
# root = "/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data3"
# # 找到所有audio_datasets.jsonl文件
# deal_files = []
# for root, dirs, files in os.walk(root):
#     for file in files:
#         if file == "audio_datasets.jsonl":
#             audio_datasets = os.path.join(root, file)
#             deal_files.append(audio_datasets)
deal_files = ["/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data3/AS/test/audio_datasets.jsonl"]


ckpt_path = config['KMEANS']['ckpt_path']  # hubert的模型
layer = config['KMEANS']['layer']  # 需要从第几层提取
max_chunk = config['KMEANS']['max_chunk']  # 每次提取的最大长度

# 初始化reader
reader = Reader(ckpt_path, layer, max_chunk)

for audio_datasets in deal_files:


    with open(audio_datasets, "r") as f:
        lines = f.readlines()
    generator, num = get_path_iterator(lines)
    iterator = generator()

    feat_path = audio_datasets.replace("audio_datasets.jsonl", "hubert_{}layer.npy".format(layer))
    leng_path = audio_datasets.replace("audio_datasets.jsonl", "hubert_{}layer.len".format(layer))

    if os.path.exists(feat_path):
        os.remove(feat_path)
    if os.path.exists(leng_path):
        os.remove(leng_path)

    feat_f = NpyAppendArray(feat_path)

    update_file = []

    with open(leng_path, "w") as leng_f:
        for item in tqdm.tqdm(iterator, total=num):
            path = item["source"]
            feat = reader.get_feats(path)
            feat_f.append(feat.cpu().detach().numpy())
            source_len = len(feat)
            leng_f.write(f"{source_len}\n")
            item["source_len"] = source_len
            update_file.append(item)

    os.remove(audio_datasets)
    with open(audio_datasets, "w") as f:
        for item in update_file:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(feat_path, "{} finished successfully".format(audio_datasets))


