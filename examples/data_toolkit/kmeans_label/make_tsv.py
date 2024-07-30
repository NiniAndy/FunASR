import numpy as np


path = "/nvme1/zhuang/dataset/data_thchs30/data_thchs30_mfcc_label_first.npz"
train_dev_test = np.load(path, allow_pickle=True)

train_file_path = train_dev_test['train_filepath'][:, [0]].tolist()
root = train_file_path[0][0][:61] + '\n'
with open('/nvme1/zhuang/dataset/data_thchs30/data/train.tsv', 'w') as f:
    f.writelines(root)
    for i in train_file_path:
        audio_path = i[0].split('/')[-1] + '\n'
        f.writelines(audio_path)



dev_file_path = train_dev_test['dev_filepath'][:, [0]].tolist()
root = dev_file_path[0][0][:58] + '\n'
with open('/nvme1/zhuang/dataset/data_thchs30/data/valid.tsv', 'w') as f:
    f.writelines(root)
    for i in dev_file_path:
        audio_path = i[0].split('/')[-1] + '\n'
        f.writelines(audio_path)

print ("OK")