import numpy as np
from config import _C as config
import torchaudio
from tqdm import tqdm


def mk2lst(lab_path):
    lab_lst = []
    with open(lab_path, "r") as f:
        for lines in f:
            lines_lst = lines.rstrip().split("\t")
            for labels in lines_lst:
                lab_lst.append(labels.split(" "))
    return lab_lst

def len2lst(len_path):
    len_lst = []
    with open(len_path, "r") as f:
        for lines in f:
            len_lst.append(lines.rstrip())
    return len_lst



config = config.KMEANS
train_dev_test = np.load(config.data_path)
# # 训练集参数
train_file_path = train_dev_test['train_filepath'][:, [0]].tolist()
train_lab_lst = mk2lst(config.lab_path)
train_len_lst = len2lst(config.train_data_mfcc_len)

# 验证集参数
dev_file_path = train_dev_test['dev_filepath'][:, [0]].tolist()
dev_lab_lst = mk2lst(config.dev_lab_path)
dev_len_lst = len2lst(config.dev_data_mfcc_len)


train = []
for i in tqdm(range(len(train_file_path))):
    sub_lst = []
    path = train_file_path[i][0]
    sub_lst.append(path)
    label = train_lab_lst[i]
    sub_lst.append(label)
    train.append(sub_lst)
train = np.array(train)


dev = []
for i in tqdm(range(len(dev_file_path))):
    sub_lst = []
    path = dev_file_path[i][0]
    sub_lst.append(path)
    label = dev_lab_lst[i]
    sub_lst.append(label)
    dev.append(sub_lst)
dev = np.array(dev)

save_path = "/nvme1/zhuang/dataset/data_thchs30/data_thchs30_mfcc_label_first"
np.savez(save_path,
         train_filepath = train,
         dev_filepath= dev)

print ("ok")
