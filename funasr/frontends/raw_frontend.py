# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from espnet/espnet.
from typing import Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as kaldi
from torch.nn.utils.rnn import pad_sequence

import funasr.frontends.eend_ola_feature as eend_ola_feature
from funasr.register import tables


@tables.register("frontend_classes", "RawFrontend")
class RawFrontend(nn.Module):
    """Conventional frontend structure for ASR."""

    def __init__(
        self,
        fs: int = 16000,
        normalize: bool = True,
        upsacle_samples: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.fs = fs
        self.normalize = normalize
        self.upsacle_samples = upsacle_samples


    def normalized_wave(self, x, eps=1e-10):
        # 使用 torch.max() 和 torch.abs() 处理张量
        # torch.max() 用于取得绝对值的最大值，这里应用 torch.max() 在张量的所有元素上
        # 这里我们需要保证不改变原始的张量尺寸，所以使用 keepdim=True
        max_val = torch.max(torch.abs(x), dim=1, keepdim=True)[0]
        x = x / (max_val + eps)
        return x

    def forward(
        self,
        input: torch.Tensor,
        input_lengths,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            waveform_length = input_lengths[i]
            waveform = input[i][:waveform_length]
            if self.upsacle_samples:
                waveform = waveform * (1 << 15)

            waveform = waveform.unsqueeze(0)  # (1 ,T)
            if self.normalize:
                waveform = self.normalized_wave(waveform)
            waveform = waveform.reshape(-1, 1) # (T, 1)
            feat_length = waveform.size(0)

            feats.append(waveform) # (B, T)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens)
        if batch_size == 1:
            feats_pad = feats[0][None, :, :]
        else:
            feats_pad = pad_sequence(feats, batch_first=True, padding_value=0.0)
        # feats_pad: (B, T, 1)
        # feats_lens: (B, )
        return feats_pad, feats_lens

   