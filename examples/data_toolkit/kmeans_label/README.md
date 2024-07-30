# K-means特征提取器

要求的环境：  
```angular2html
fairseq                       0.12.2
npy-append-array              0.9.16
yacs                          0.1.8
```

[Hubert下载]: https://github.com/facebookresearch/fairseq/tree/main/examples/hubert

请按以下步骤完成：

1. feature extraction
2. k-means clustering
3. k-means application


## Data preparation

`*.npy` 文件文件路径:
```
<audio-path-1>
<audio-path-2>
...
```


## Feature extraction

### MFCC feature
To extract 39-D mfcc+delta+ddelta features for the 1st iteration HUBERT training, run:
```sh
python dump_mfcc_feature.py 
```
Features would be saved at `$/_mfcc.{npy,len}`.


### HUBERT feature
To extract features from the `${layer}`-th transformer layer of a trained
HUBERT model saved at `${ckpt_path}`, run:
```sh
python dump_hubert_feature.py
```
Features would also be saved at `$/_hubert.{npy,len}`.
这里需要在config.py里面加入保存的路径

- if out-of-memory, decrease the chunk size with `--max_chunk`


## K-means clustering
To fit a k-means model with `${n_clusters}` clusters on 10% of the `${split}` data, run
```sh
python learn_kmeans.py
```
This saves the k-means model to `${km_path}`.

- set `--precent -1` to use all data
- more kmeans options can be found with `-h` flag


## K-means application
To apply a trained k-means model `${km_path}` to obtain labels for all the dataset, run
```sh
python dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
```
This would extract labels for the all the dataset
and dump them to `${lab_dir}/${split}_${rank}_${shard}.km`

最后根据生成的标签使用`make_new_dataset.py`获得初始数集集