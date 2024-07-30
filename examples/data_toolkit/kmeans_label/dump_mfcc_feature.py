import numpy as np
import tqdm
from npy_append_array import NpyAppendArray
from mfcc_extractor import load_audio, ExtractFbankFeature
from config import _C as config
import os





class Reader(object):
    def __init__(self, config):
        self.extract_feature = ExtractFbankFeature(config)

    def get_feats(self, path):
        waveform, simple_rate = load_audio(path)
        feature = self.extract_feature(waveform, simple_rate)
        return feature.transpose(0, 1).contiguous()


def get_path_iterator(file_path):
        def iterate():
            for audio_path in file_path:
                yield f"{audio_path}"
        return iterate, len(file_path)



##############  main #######################

reader = Reader(config)

train_dev_test = np.load(config.KMEANS.data_path)
##########  train   ############
file_path = train_dev_test['train_100'][:].tolist() \
            + train_dev_test['train_360'][:].tolist() \
            + train_dev_test['train_500'][:].tolist()


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
        feat_f.append(feat.cpu().numpy())
        leng_f.write(f"{len(feat)}\n")
print (feat_path, "train finished successfully")

##########  dev   ############
file_path = train_dev_test['train_100'][:].tolist()
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
        feat_f.append(feat.cpu().numpy())
        leng_f.write(f"{len(feat)}\n")
print (feat_path, "dev finished successfully")
