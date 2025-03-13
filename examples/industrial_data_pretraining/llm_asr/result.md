# LLM-ASR aishell-1 Result

## Training Config

### llm_asr_whisper_linear_qwen_aishell1_wo_specaug

-  Results (CER)

| dev | test  | 
|:---:|:-----:|
| --- | 5.08  |

```yaml
model: LLMASR
model_conf:
  lsm_weight: 0.1
  length_normalized_loss: true
audio_encoder: /ssd/zhuang/code/LLM/whisper-large-v3-ms/Whisper-large-v3
audio_encoder_conf:
  hub: ms
  freeze: true
llm: Qwen1.5-7b-chat
llm_conf:
  hub: hf
  freeze: true
  init_param_path: /ssd/zhuang/code/LLM/Qwen1.5-7B-Chat/
audio_adaptor: Linear
audio_adaptor_conf:
  downsample_rate: 5
  llm_dim: 4096
  encoder_dim: 1280
frontend: WhisperFrontend
frontend_conf:
  fs: 16000
  whisper_model: large-v3
  do_pad_trim: true
  permute: true
train_conf:
  accum_grad: 1
  grad_clip: 5
  max_epoch: 10
  keep_nbest_models: 10
  log_interval: 300
optim: adamw
optim_conf:
  lr: 0.0001
  weight_decay: 0.0
scheduler: warmuplr
scheduler_conf:
  warmup_steps: 1500
dataset: AudioLLMQwenAudioDataset
dataset_conf:
  index_ds: IndexDSJsonl
  batch_sampler: CustomDistributedBatchSampler
  batch_type: example
  batch_size: 2
  max_token_length: 3000
  shuffle: true
  num_workers: 4
  preprocessor_text: TextPreprocessRemovePunctuation
  audio_adaptor_downsample_rate: ${audio_adaptor_conf.downsample_rate}
  audio_encoder_downsample_rate: 2
tokenizer: HuggingfaceTokenizer
tokenizer_conf:
  unk_symbol: <unk>
  init_param_path: /ssd/zhuang/code/LLM/Qwen1.5-7B-Chat/
train_data_set_list: /ssd/zhuang/code/FunASR/examples/aishell/DATA/data/train/audio_datasets.jsonl
valid_data_set_list: /ssd/zhuang/code/FunASR/examples/aishell/DATA/data/dev/audio_datasets.jsonl
output_dir: /ssd/zhuang/code/FunASR/examples/industrial_data_pretraining/llm_asr/exp/llm_asr_whisper_qwen_aishell1_exp
device: cpu
```


### llm_asr_whisper_linear_qwen_aishell1

-  Results (CER)

| dev  | test | 
|:----:|:----:|
| 4.36 | 4.88 |

```yaml
model: LLMASR
model_conf:
  lsm_weight: 0.1
  length_normalized_loss: true
audio_encoder: /ssd/zhuang/code/LLM/whisper-large-v3-ms/Whisper-large-v3
audio_encoder_conf:
  hub: ms
  freeze: true
llm: Qwen1.5-7b-chat
llm_conf:
  hub: hf
  freeze: true
  init_param_path: /ssd/zhuang/code/LLM/Qwen1.5-7B-Chat/
audio_adaptor: Linear
audio_adaptor_conf:
  downsample_rate: 5
  llm_dim: 4096
  encoder_dim: 1280
frontend: WhisperFrontend
frontend_conf:
  fs: 16000
  whisper_model: large-v3
  do_pad_trim: true
  permute: true
specaug: SpecAugLFR
specaug_conf:
  apply_time_warp: false
  time_warp_window: 5
  time_warp_mode: bicubic
  apply_freq_mask: true
  freq_mask_width_range:
  - 0
  - 30
  lfr_rate: 6
  num_freq_mask: 1
  apply_time_mask: true
  time_mask_width_range:
  - 0
  - 12
  num_time_mask: 1
train_conf:
  accum_grad: 1
  grad_clip: 5
  max_epoch: 20
  keep_nbest_models: 10
  log_interval: 300
optim: adamw
optim_conf:
  lr: 0.0001
  weight_decay: 0.0
scheduler: warmuplr
scheduler_conf:
  warmup_steps: 3000
dataset: AudioLLMQwenAudioDataset
dataset_conf:
  index_ds: IndexDSJsonl
  batch_sampler: CustomDistributedBatchSampler
  batch_type: example
  batch_size: 2
  max_token_length: 3000
  shuffle: true
  num_workers: 4
  preprocessor_text: TextPreprocessRemovePunctuation
  audio_adaptor_downsample_rate: ${audio_adaptor_conf.downsample_rate}
  audio_encoder_downsample_rate: 2
tokenizer: HuggingfaceTokenizer
tokenizer_conf:
  unk_symbol: <unk>
  init_param_path: /ssd/zhuang/code/LLM/Qwen1.5-7B-Chat/
train_data_set_list: /ssd/zhuang/code/FunASR/examples/aishell/DATA/data/train/audio_datasets.jsonl
valid_data_set_list: /ssd/zhuang/code/FunASR/examples/aishell/DATA/data/dev/audio_datasets.jsonl
output_dir: /ssd/zhuang/code/FunASR/examples/industrial_data_pretraining/llm_asr/exp/llm_asr_whisper_qwen_aishell1_exp
device: cpu
```