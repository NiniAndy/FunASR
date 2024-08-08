import logging

import torch

from funasr.models.transformer.model_multi_emb import MultiTransformer
from funasr.register import tables


@tables.register("model_classes", "MultiConformer")
class MultiConformer(MultiTransformer):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
