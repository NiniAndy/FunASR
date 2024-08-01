# 用于Debug的形参



## transformer training

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



## whisper_qwen_linear_finetune

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

## ar_transformer_traning

```
# 用<acc>替换了<sos>
--config-path
/ssd/zhuang/code/FunASR/examples/aishell/transformer/conf
--config-name
transformer_12e_6d_2048_256.yaml
++train_data_set_list=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data2/ES/Southwestern/train/audio_datasets.jsonl
++valid_data_set_list=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data2/ES/Southwestern/dev/audio_datasets.jsonl
++tokenizer_conf.token_list=/ssd/zhuang/code/FunASR/examples/aishell/paraformer/exp/speech_paraformer_asr_nat-aishell1-pytorch/tokens.txt
++tokenizer_conf.add_special_token_list=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data2/zh_token_list/char/dialects.txt
++frontend_conf.cmvn_file=/ssd/zhuang/code/FunASR/examples/aishell/paraformer/exp/speech_paraformer_asr_nat-aishell1-pytorch/am.mvn
++output_dir=/ssd/zhuang/code/FunASR/examples/aishell/transformer/exp/debug
++scheduler=partition_warmuplr
++text_language_vocab_path=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data2/zh_token_list/char/dialects.txt
```

## ar_transformer_test

```
# 用<acc>替换了<sos>的测试
--config-path=/ssd/zhuang/code/FunASR/examples/kespeech/conformer/exp/baseline_conformer_12e_6d_2048_256_zh_char_Whole_withAR_prompt
--config-name=config.yaml
++init_param=/ssd/zhuang/code/FunASR/examples/kespeech/conformer/exp/baseline_conformer_12e_6d_2048_256_zh_char_Whole_withAR_prompt/model.pt.avg10
++tokenizer_conf.token_list=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data2/zh_token_list/char/tokens.txt
++tokenizer_conf.add_special_token_list=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data2/zh_token_list/char/dialects.txt
++frontend_conf.cmvn_file=/ssd/zhuang/code/FunASR/examples/aishell/paraformer/exp/speech_paraformer_asr_nat-aishell1-pytorch/am.mvn
++output_dir=/ssd/zhuang/code/FunASR/examples/aishell/transformer/exp/debug
++text_language_vocab_path=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data2/zh_token_list/char/dialects.txt
++input=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data2/AS/test/audio_datasets.jsonl
++output_dir=/ssd/zhuang/code/FunASR/examples/kespeech/conformer/exp/baseline_conformer_12e_6d_2048_256_zh_char_Whole_withAR_prompt
++device=cuda
++ncpu=1
++disable_log=true
++batch_size=1
```

## conformer_ar_training

```
# 口音识别
--config-path
/ssd/zhuang/code/FunASR/examples/kespeech/conformer_ar/conf
--config-name
conformer_12e_6d_2048_256.yaml
++train_data_set_list=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data2/ES/Southwestern/train/audio_datasets.jsonl
++valid_data_set_list=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data2/ES/Southwestern/dev/audio_datasets.jsonl
++tokenizer_conf.token_list=/ssd/zhuang/code/FunASR/examples/aishell/paraformer/exp/speech_paraformer_asr_nat-aishell1-pytorch/tokens.txt
++tokenizer_conf.add_special_token_list=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data2/zh_token_list/char/dialects.txt
++frontend_conf.cmvn_file=/ssd/zhuang/code/FunASR/examples/aishell/paraformer/exp/speech_paraformer_asr_nat-aishell1-pytorch/am.mvn
++output_dir=/ssd/zhuang/code/FunASR/examples/kespeech/conformer_ar/exp/debug
++text_language_vocab_path=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data2/zh_token_list/char/dialects.txt
```











# 模型的配置

## Encoder

### TransformerEncoder

```
encoder: TransformerEncoder
encoder_conf:
    output_size: 256    # dimension of attention
    attention_heads: 4
    linear_units: 2048  # the number of units of position-wise feed forward
    num_blocks: 12      # the number of encoder blocks
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    attention_dropout_rate: 0.0
    input_layer: conv2d # encoder architecture type
    normalize_before: true
```

### ConformerEncoder

```
encoder_conf:
    output_size: 256    # dimension of attention
    attention_heads: 4
    linear_units: 2048  # the number of units of position-wise feed forward
    num_blocks: 12      # the number of encoder blocks
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    attention_dropout_rate: 0.0
    input_layer: conv2d # encoder architecture type
    normalize_before: true
    pos_enc_layer_type: rel_pos
    selfattention_layer_type: rel_selfattn
    activation_type: swish
    macaron_style: true
    use_cnn_module: true
    cnn_module_kernel: 15
```

### Encoder with Wav2Vec2.0 or Hubert feature extractor

```
encoder: ConformerEncoder
encoder_conf:
    output_size: 256    # dimension of attention
    attention_heads: 4
    linear_units: 2048  # the number of units of position-wise feed forward
    num_blocks: 12      # the number of encoder blocks
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    attention_dropout_rate: 0.0
    input_layer: hubert # encoder architecture type
    normalize_before: true
    pos_enc_layer_type: rel_pos
    selfattention_layer_type: rel_selfattn
    activation_type: swish
    macaron_style: true
    use_cnn_module: true
    cnn_module_kernel: 15
    interctc_layer_idx: [ 6, 9 ]
    interctc_use_conditioning: true
    conv_conf:
      conv_bias: False
      conv_dim: [ 512, 512, 512, 512, 512, 512, 512 ]
      conv_kernel: [ 10, 3, 3, 3, 3, 2, 2 ]
      conv_stride: [ 5, 2, 2, 2, 2, 2, 2 ]
      conv_activation: "gelu"
    feat_extract_conf:
      feat_extract_norm: "group"
      num_feat_extract_layers: 7
      feat_proj_layer_norm: True
      feat_proj_dropout: 0.0
```





## Decoder



### TransformerDecoder

```
decoder: TransformerDecoder
decoder_conf:
    attention_heads: 4
    linear_units: 2048
    num_blocks: 6
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    self_attention_dropout_rate: 0.0
    src_attention_dropout_rate: 0.0
```



