import numpy as np
from sklearn.cluster import MiniBatchKMeans
from config import _C as config
import os
import joblib



def load_feature(data, percent):
    assert percent <= 1.0
    feat_path, leng_path = data[0], data[1]
    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    nsample = int(np.ceil(len(lengs) * percent))
    indices = np.random.choice(len(lengs), nsample, replace=False)
    feat = np.load(feat_path, mmap_mode="r")
    sampled_feat = np.concatenate([feat[offsets[i]: offsets[i] + lengs[i]] for i in indices], axis=0)
    print (f"sampled {nsample} utterances, {len(sampled_feat)} frames ")
    print (f"loaded feature with dimension {sampled_feat.shape}")
    return sampled_feat


def get_km_model( n_clusters, max_iter, batch_size,
                  tol=0.0, max_no_improvement=10, n_init=50, reassignment_ratio=0.0, init="k-means++"):
    return MiniBatchKMeans( n_clusters=n_clusters,
                            init=init,
                            max_iter=max_iter,
                            batch_size=batch_size,
                            verbose=1,
                            compute_labels=False,
                            tol=tol,
                            max_no_improvement=max_no_improvement,
                            init_size=None,
                            n_init=n_init,
                            reassignment_ratio=reassignment_ratio)





################ main  #################
config  = config.KMEANS
percent = config.percent
train_data = [config.train_data_mfcc, config.train_data_mfcc_len]

np.random.seed(0)
feat = load_feature(train_data, percent)
km_model = get_km_model(config.n_clusters, config.max_iter, config.batch_size)
km_model.fit(feat)
joblib.dump(km_model, config.km_path)

inertia = -km_model.score(feat) / len(feat)
print ("total intertia: ", inertia)
print ("finished successfully")
