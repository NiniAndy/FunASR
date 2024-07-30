from yacs.config import CfgNode as CN
import time
import os
root_dir = os.getcwd()
date = time.strftime("%Y-%m-%d", time.localtime())

# 这一部分是默认参数，临时修改参数请到yml文件里指定
_C = CN()

# 音频数据设置，窗函数可以选择 hamming hann blackman bartlett, 窗长单位为s
_C.AUDIO = CN()
_C.AUDIO.sample_rate = 16000
_C.AUDIO.window_size = .02
_C.AUDIO.window_stride = .01
_C.AUDIO.window = 'hamming'

# 特征提取
_C.FEATURE = CN()
_C.FEATURE.spectrum_normalize = True  # 谱归一化
_C.FEATURE.feature = 'fbank'  # stft, fbank
# fbank参数
_C.FEATURE.delta_order = 2
_C.FEATURE.num_mel_bins = 13
_C.FEATURE.delta_window_size = 1


# Kmeans的参数
_C.KMEANS = CN()
_C.KMEANS.n_clusters = 150
_C.KMEANS.percent = 0.3
_C.KMEANS.max_iter = 1000
_C.KMEANS.batch_size = 10000
# 文件夹的位置
_C.KMEANS.km_path = "/data/NAS_PLUS/zhuang/CodeBase/008_simple_kmeans/save/librispeech_kmean_model"  # kmeans模型保存路径
_C.KMEANS.data_path = "/data/NAS_PLUS/zhuang/dataset/data_librispeech/data_librispeech_train.npz"
# train保存路径
_C.KMEANS.train_data_mfcc = r"/nvme0/zhuang/data_librispeech_train_mfcc.npy"
_C.KMEANS.train_data_mfcc_len = r"/data/NAS_PLUS/zhuang/CodeBase/008_simple_kmeans/save/data_librispeech_train_mfcc.len"
# val保存路径
_C.KMEANS.dev_data_mfcc = r"/data/NAS_PLUS/zhuang/CodeBase/008_simple_kmeans/save/data_librispeech_train_val_mfcc.npy"
_C.KMEANS.dev_data_mfcc_len = r"/data/NAS_PLUS/zhuang/CodeBase/008_simple_kmeans/save/data_librispeech_train_val_mfcc.len"
# Hubert的参数
# _C.KMEANS.ckpt_path= ""
_C.KMEANS.ckpt_path= "/data/NAS_PLUS/zhuang/CodeBase/008_simple_kmeans/save/hubert_base_ls960.pt"
_C.KMEANS.layer = 6
_C.KMEANS.max_chunk = 1600000
# 输出label路径
_C.KMEANS.lab_path = r"/data/NAS_PLUS/zhuang/CodeBase/008_simple_kmeans/save/data_librispeech_label.km"
_C.KMEANS.dev_lab_path = r"/data/NAS_PLUS/zhuang/CodeBase/008_simple_kmeans/save/data_librispeech_dve_label.km"
_C.KMEANS.new_dataset = ""