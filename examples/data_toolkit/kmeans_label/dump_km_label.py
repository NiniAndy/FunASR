import os
import sys
import numpy as np
import joblib
import torch
import tqdm
import yaml

# 读取配置文件
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


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
root = "/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data3"
feature_name = "hubert_9layer"
# # 找到所有audio_datasets.jsonl文件
# deal_files = []
# for root, dirs, files in os.walk(root):
#     for file in files:
#         if file == "audio_datasets.jsonl":
#             audio_datasets = os.path.join(root, file)
#             deal_files.append(audio_datasets)

deal_files = ["/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data3/AS/test/audio_datasets.jsonl"]
km_model = config["KMEANS"]["km_path"]
apply_kmeans = ApplyKmeans(km_model)


for file in deal_files:
    print ("processing file: ", file)
    feature_path = file.replace("audio_datasets.jsonl", "{}.npy".format(feature_name))
    feature_leng = file.replace("audio_datasets.jsonl", "{}.len".format(feature_name))
    generator, num = get_feat_iterator([feature_path, feature_leng])
    iterator = generator()
    with open(file.replace("audio_datasets.jsonl", "kmeans_label"), "w") as f:
        for feat in tqdm.tqdm(iterator, total=num):
            lab = apply_kmeans(feat).tolist()
            f.write(" ".join(map(str, lab)) + "\n")
print ("finished successfully")



