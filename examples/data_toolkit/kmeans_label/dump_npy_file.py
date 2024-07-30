import os
import numpy as np
from tqdm import tqdm

# root_path = "/data/NAS_PLUS/asr_datasets/LibriSpeech/train-clean-100/"
#
# train_100 = []
# for root, name, file in tqdm(os.walk(root_path)):
#     if  len(file) != 0 :
#         for file_name in file:
#             if file_name.endswith("txt"):
#                 continue
#             else:
#                 train_100.append(os.path.join(root, file_name))
#
#
# root_path = "/data/NAS_PLUS/asr_datasets/LibriSpeech/train-clean-360/"
#
# train_360 = []
# for root, name, file in tqdm(os.walk(root_path)):
#     if  len(file) != 0 :
#         for file_name in file:
#             if file_name.endswith("txt"):
#                 continue
#             else:
#                 train_360.append(os.path.join(root, file_name))
#
#
# root_path = "/data/NAS_PLUS/asr_datasets/LibriSpeech/train-other-500/"
# train_500 = []
# for root, name, file in tqdm(os.walk(root_path)):
#     if len(file) != 0:
#         for file_name in file:
#             if file_name.endswith("txt"):
#                 continue
#             else:
#                 train_500.append(os.path.join(root, file_name))
#
#
#
# save_path = "/data/NAS_PLUS/zhuang/dataset/data_librispeech/data_librispeech_train.npz"
# np.savez(save_path,
#          train_100 = train_100,
#          train_360 = train_360,
#          train_500 = train_500)


# save_path = "/data/NAS_PLUS/zhuang/dataset/data_librispeech/data_librispeech_train.npz"
# a = np.load(save_path)
# b = a['train_100'][:].reshape(-1, 1).tolist() + a['train_360'][:].reshape(-1, 1).tolist() +  a['train_500'][:].reshape(-1, 1).tolist()
#
# print (b)
# import torchaudio
# c, d = torchaudio.load(b[0])


# root_path = "/data/NAS_PLUS/asr_datasets/LibriSpeech/dev-clean/"
#
# dev_clean = []
# for root, name, file in tqdm(os.walk(root_path)):
#     if  len(file) != 0 :
#         for file_name in file:
#             if file_name.endswith("txt"):
#                 continue
#             else:
#                 dev_clean.append(os.path.join(root, file_name))
#
#
# root_path = "/data/NAS_PLUS/asr_datasets/LibriSpeech/dev-other/"
#
# dev_other = []
# for root, name, file in tqdm(os.walk(root_path)):
#     if  len(file) != 0 :
#         for file_name in file:
#             if file_name.endswith("txt"):
#                 continue
#             else:
#                 dev_other.append(os.path.join(root, file_name))
#
# save_path = "/data/NAS_PLUS/zhuang/dataset/data_librispeech/data_librispeech_dev.npz"
# np.savez(save_path,
#          dev_clean = dev_clean,
#          dev_other = dev_other)


# save_path = "/data/NAS_PLUS/zhuang/dataset/data_librispeech/data_librispeech_dev.npz"
# a = np.load(save_path)
# b = a['dev_clean'][:].reshape(-1, 1).tolist() + a['dev_other'][:].reshape(-1, 1).tolist()
#
# print (b)
# import torchaudio
# c, d = torchaudio.load(b[0])


dev_path = "/data/NAS_PLUS/zhuang/dataset/data_librispeech/data_librispeech_dev.npz"
train_path = "/data/NAS_PLUS/zhuang/dataset/data_librispeech/data_librispeech_train.npz"
train = np.load(train_path)
train_lst = train['train_100'][:].reshape(-1, 1).tolist() \
            + train['train_360'][:].reshape(-1, 1).tolist() \
            + train['train_500'][:].reshape(-1, 1).tolist()

dev = np.load(dev_path)
dev_lst = dev['dev_clean'][:].reshape(-1, 1).tolist() \
            + dev['dev_other'][:].reshape(-1, 1).tolist()

save_path = "/data/nas/zhuang/dataset/data_librispeech/train_dev_test.npz"
np.savez(save_path,
         train_filepath=train_lst,
         dev_filepath=dev_lst,
         test_filepath=[]
         )

print ("ok")