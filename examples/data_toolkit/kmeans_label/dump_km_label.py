import os
import sys
import numpy as np
import joblib
import torch
import tqdm
from config import _C as config


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = ( x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm)
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = ( (x ** 2).sum(1, keepdims=True) - 2 * np.matmul(x, self.C_np) + self.Cnorm_np)
            return np.argmin(dist, axis=1)




def get_feat_iterator(file_lst):
    feat_path = file_lst[0]
    leng_path = file_lst[1]
    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    def iterate():
        feat = np.load(feat_path, mmap_mode="r")
        assert feat.shape[0] == (offsets[-1] + lengs[-1])
        for offset, leng in zip(offsets, lengs):
            yield feat[offset: offset + leng]

    return iterate, len(lengs)



#############  main  ################
config = config.KMEANS
dev = False
if dev:
    file_lst = [config.dev_data_mfcc, config.dev_data_mfcc_len]
    lab_path = config.dev_lab_path
else:
    file_lst = [config.train_data_mfcc, config.train_data_mfcc_len]
    lab_path = config.lab_path
apply_kmeans = ApplyKmeans(config.km_path)
generator, num = get_feat_iterator(file_lst)
iterator = generator()
with open(lab_path, "w") as f:
    for feat in tqdm.tqdm(iterator, total=num):
        # feat = torch.from_numpy(feat).cuda()
        lab = apply_kmeans(feat).tolist()
        f.write(" ".join(map(str, lab)) + "\n")