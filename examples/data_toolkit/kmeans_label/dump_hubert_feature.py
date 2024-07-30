import logging
import math
import os
import sys
import numpy as np
import fairseq
import soundfile as sf
import torch
import torch.nn.functional as F
import tqdm
from npy_append_array import NpyAppendArray
from config import _C as config
from mfcc_extractor import load_audio


class Reader(object):
    def __init__(self,ckpt_path, layer, max_chunk=1600000):
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



def get_path_iterator(file_path):
        def iterate():
            for audio_path in file_path:
                yield f"{audio_path}"
        return iterate, len(file_path)



##################  main  #####################
reader = Reader(config.KMEANS.ckpt_path, config.KMEANS.layer, config.KMEANS.max_chunk)
train_dev_test = np.load(config.KMEANS.data_path)

##########  train   ############
file_path = train_dev_test['train_filepath'][:, [0]].tolist()
generator, num = get_path_iterator(file_path)
iterator = generator()

feat_path = config.KMEANS.train_data_mfcc
leng_path = config.KMEANS.train_data_mfcc_len

if os.path.exists(feat_path):
    os.remove(feat_path)

feat_f = NpyAppendArray(feat_path)
with open(leng_path, "w") as leng_f:
    for path in tqdm.tqdm(iterator, total=num):
        feat = reader.get_feats(path)
        feat_f.append(feat.cpu().detach().numpy())
        leng_f.write(f"{len(feat)}\n")
print (feat_path, "train finished successfully")


##########  dev   ############
file_path = train_dev_test['dev_filepath'][:, [0]].tolist()
generator, num = get_path_iterator(file_path)
iterator = generator()

feat_path = config.KMEANS.dev_data_mfcc
leng_path = config.KMEANS.dev_data_mfcc_len

if os.path.exists(feat_path):
    os.remove(feat_path)

feat_f = NpyAppendArray(feat_path)
with open(leng_path, "w") as leng_f:
    for path in tqdm.tqdm(iterator, total=num):
        feat = reader.get_feats(path)
        feat_f.append(feat.cpu().detach().numpy())
        leng_f.write(f"{len(feat)}\n")
print (feat_path, "dev finished successfully")