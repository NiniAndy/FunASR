# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer encoder definition."""

from typing import List
from typing import Optional
from typing import Tuple

import torch
from torch import nn
import logging

from funasr.models.transformer.attention import MultiHeadedAttention
from funasr.models.transformer.embedding import PositionalEncoding
from funasr.models.transformer.layer_norm import LayerNorm

from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.models.transformer.positionwise_feed_forward import PositionwiseFeedForward
from funasr.models.transformer.utils.repeat import repeat
from funasr.models.transformer.encoder import EncoderLayer

from funasr.register import tables



@tables.register("encoder_classes", "TextBertEncoder")
class TransformerEncoder(nn.Module):

    def __init__(
        self,
        input_vocal_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 4,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        super().__init__()
        self._output_size = output_size
        attention_dim = output_size

        self.embed = torch.nn.Sequential(
            torch.nn.Embedding(input_vocal_size, attention_dim),
            pos_enc_class(attention_dim, positional_dropout_rate),
        )

        self.normalize_before = normalize_before

        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
        )

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                MultiHeadedAttention(attention_heads, output_size, attention_dropout_rate),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)


    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        xs_pad = self.embed(xs_pad)

        xs_pad, masks = self.encoders(xs_pad, masks)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)
        olens = masks.squeeze(1).sum(1)

        return xs_pad, olens, None
