#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import os

from typing import Dict

from .layers import LoRALayer, Linear, ConvLoRA, Embedding


def mark_only_lora_as_trainable(
    model: nn.Module, bias: str = "none", lora_exception: list = []
) -> None:
    """
    lora_exception: list控制可以训练的层
    """
    # LoRA内部的参数全部设置为可训练
    for n, p in model.named_parameters():
        if "lora_" not in n and "cif" not in n:
            p.requires_grad = False
    # 控制其他可以训练的层
    if len(lora_exception) > 0:
        for exception in lora_exception:
            for n, p in model.named_parameters():
                if exception in n:
                    p.requires_grad = True
    # 控制bias是否可以训练
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = "none") -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == "none":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k}
    elif bias == "all":
        return {
            k: my_state_dict[k] for k in my_state_dict if "lora_" in k or "bias" in k
        }
    elif bias == "lora_only":
        to_return = {}
        for k in my_state_dict:
            if "lora_" in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def lora_summary(model: nn.Module) -> str:
    message = "Lora train able params:\n"
    train_able_params_list = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            train_able_params_list.append(name)
    for name in train_able_params_list:
        message += f"    {name}\n"
    return message


def loar_wrapper(
    model: nn.Module, substitute_dict: dict, lora_kwargs: dict
) -> nn.Module:
    """
    substitute_list: list, 代表需要替换的层, 如['encoder.q', 'decoder.k', 'decoder.v', 'decoder.out_proj]
    在conformer_encoder中有几个部分可以更改:{a: b, c: d, e: f}
    {   encoder:    {forward_macaron: [w_1, w_2]},
                    {self_attn: [linear_q, linear_k, linear_v, linear_out, linear_pos]},
                    {feed_forward: [w_1, w_2]}},
        decoder:    {self_attn: [linear_q, linear_k, linear_v, linear_out],
                    {src_attn: [linear_q, linear_k, linear_v, linear_out],}},
                    {feed_forward: [w_1, w_2]}
    }
    """
    # 检查model里是否包含substitute_dict的模型
    for module_name, M_module_name in substitute_dict.items():
        module = getattr(model, module_name)
        # 如果module本身是nn.linear, nn.conv1d, nn.conv2d, nn.embedding, 则直接替换
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.Embedding):
            replace_layer_recursive(module, module_name, lora_kwargs)
        else:
            for M_name, M_substitute_list in M_module_name.items():
                if M_substitute_list is None:
                    replace_layer_recursive(module, M_name, lora_kwargs)
                else:
                    for sub_name in M_substitute_list:
                        replace_layer_recursive(module, sub_name, lora_kwargs)
    return model


def replace_layer_recursive(module, target_layer_name, lora_kwargs, parent_name=""):
    # 遍历当前模块的所有子模块
    for name, child in list(module.named_children()):
        # 构建当前层的全路径名称
        if parent_name:
            full_name = f"{parent_name}.{name}"
        else:
            full_name = name

        # 检查当前子模块是否是需要替换的Linear层
        if isinstance(child, nn.Linear) and name == target_layer_name:
            in_features = child.in_features
            out_features = child.out_features
            replace_layer = Linear(in_features, out_features, **lora_kwargs)
            setattr(module, name, replace_layer)
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if local_rank == 0:
                print(f"Replacing layer: {full_name} with new LoRA linear layer")

        elif isinstance(child, nn.Conv1d) and name == target_layer_name:
            in_channels = child.in_channels
            out_channels = child.out_channels
            kernel_size = child.kernel_size[0]
            stride = child.stride[0]
            padding = child.padding[0]
            groups = child.groups
            bias = child.bias is not None
            replace_layer = ConvLoRA(
                nn.Conv1d,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups,
                bias,
                **lora_kwargs,
            )
            setattr(module, name, replace_layer)
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if local_rank == 0:
                print(f"Replacing layer: {full_name} with new LoRA Conv2d layer")

        elif isinstance(child, nn.Conv2d) and name == target_layer_name:
            in_channels = child.in_channels
            out_channels = child.out_channels
            kernel_size = child.kernel_size
            replace_layer = ConvLoRA(
                nn.Conv2d, in_channels, out_channels, kernel_size, **lora_kwargs
            )
            setattr(module, name, replace_layer)
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if local_rank == 0:
                print(f"Replacing layer: {full_name} with new LoRA Conv2d layer")

        elif isinstance(child, nn.Embedding) and name == target_layer_name:
            embedding_dim = child.embedding_dim
            num_embeddings = child.num_embeddings
            replace_layer = Embedding(num_embeddings, embedding_dim, **lora_kwargs)
            setattr(module, name, replace_layer)
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if local_rank == 0:
                print(f"Replacing layer: {full_name} with new LoRA emb layer")
        else:
            # 如果子模块不是目标层，递归调用此函数
            replace_layer_recursive(child, target_layer_name, lora_kwargs, full_name)
