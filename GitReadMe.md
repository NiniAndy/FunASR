# Git融合方法

#### 1.确认当前分支

```
git checkout mymerge
```

#### 2.更新本地代码到github版本

```
git pull origin mymerge
```

#### 3.从github中拉取所有branch信息

```
git fetch origin
```

#### 4.选择要和哪个本版进行融合

```
git merge origin/main
```

#### 5.冲突判断

#### 6.确认完成冲突合并

```
 git commit
```

#### 7.上传合并后文件

```
git push origin mymerge
```

# 一些debug用得上的形参

#### transformer training
```angular2html
--config-path /ssd/zhuang/code/FunASR/examples/aishell/transformer/conf
--config-name transformer_12e_6d_2048_256.yaml
++train_data_set_list=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data/ES/Southwestern/train/audio_datasets.jsonl
++valid_data_set_list=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data/ES/Southwestern/dev/audio_datasets.jsonl
++tokenizer_conf.token_list=/ssd/zhuang/code/FunASR/examples/aishell/paraformer/exp/speech_paraformer_asr_nat-aishell1-pytorch/tokens.txt 
++frontend_conf.cmvn_file=/ssd/zhuang/code/FunASR/examples/aishell/paraformer/exp/speech_paraformer_asr_nat-aishell1-pytorch/am.mvn 
++output_dir=/ssd/zhuang/code/FunASR/examples/aishell/transformer/exp/debug
++scheduler=partition_warmuplr
```

#### whisper_qwen_linear_finetune
```angular2html
--config-path /ssd/zhuang/code/FunASR2024/examples/industrial_data_pretraining/llm_asr/conf 
--config-name whisper_qwen_linear.yaml 
++train_data_set_list=/ssd/zhuang/code/FunASR2024/examples/librispeech/DATA/data/train-960/audio_datasets.jsonl 
++valid_data_set_list=/ssd/zhuang/code/FunASR2024/examples/librispeech/DATA/data/dev-other/audio_datasets.jsonl 
++dataset_conf.batch_size=4 
++dataset_conf.num_workers=4 
++train_conf.max_epoch=15 
++optim_conf.lr=0.0001
++output_dir=/ssd/zhuang/code/FunASR2024/examples/industrial_data_pretraining/llm_asr/exp/llm_asr_whisper_qwen_exp1
```