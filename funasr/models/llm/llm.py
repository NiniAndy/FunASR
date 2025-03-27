import os
import types
import torch
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from funasr.register import tables

# from slam_llm.utils.config_utils import generate_peft_config
# from slam_llm.utils.train_utils import print_module_size, print_model_size
# from peft import PeftModel, PeftConfig
# from torch.nn import CrossEntropyLoss

@tables.register("llm_classes", "Vicuna")
@tables.register("llm_classes", "vicuna")
class vicuna:
    def __init__(self):
        pass

    def load(**llm_config):
        init_param_path = llm_config.get("init_param_path", None)
        quantization = llm_config.get("quantization", False)
        use_cache = llm_config.get("use_cache", False)

        model = AutoModelForCausalLM.from_pretrained(
            init_param_path,
            load_in_8bit=True if quantization else None,
            device_map="auto" if quantization else None,
            use_cache=use_cache,
        )

        return model

# def setup_llm(llm, llm_config, **kwargs):
#
#     init_param_path = llm_config.get("init_param_path", None)
#     quantization = llm_config.get("quantization", False)
#     use_cache = llm_config.get("use_cache", False)
#
#     model = AutoModelForCausalLM.from_pretrained(
#         init_param_path,
#         load_in_8bit=True if quantization else None,
#         device_map="auto" if quantization else None,
#         use_cache=use_cache,
#     )

    # if (train_config.enable_fsdp or train_config.enable_ddp) and train_config.use_fast_kernels:
    #     """
    #     For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
    #     using of Flash Attention or Xformer memory-efficient kernels
    #     based on the hardware being used. This would speed up fine-tuning.
    #     """
    #     try:
    #         from optimum.bettertransformer import BetterTransformer
    #         model = BetterTransformer.transform(model)
    #     except ImportError:
    #         logger.warning("Module 'optimum' not found. Please install 'optimum' it before proceeding.")
    #
    # print_module_size(model, model_config.llm_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    #
    # # Prepare the model for int8 training if quantization is enabled
    # if train_config.quantization:
    #     model = prepare_model_for_kbit_training(model)
    #
    # if train_config.freeze_llm:  # TODO:to test offical `freeze_layers` and `num_freeze_layers`
    #     for name, param in model.named_parameters():
    #         param.requires_grad = False
    #     model.eval()
    #
    # if kwargs.get("peft_ckpt", None):  # (FIX:MZY):reload will get wrong results when decoding
    #     logger.info("loading peft_ckpt from: {}".format(kwargs.get("peft_ckpt")))
    #     model = PeftModel.from_pretrained(model=model, model_id=kwargs.get("peft_ckpt"), is_trainable=True)
    #     model.print_trainable_parameters()
    #
    # elif train_config.use_peft:
    #     logger.info("setup peft...")
    #     peft_config = generate_peft_config(train_config)
    #     model = get_peft_model(model, peft_config)
    #     model.print_trainable_parameters()
    #
    # print_module_size(model, model_config.llm_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    # return model