import logging

import torch

from funasr.models.transformer.model_arasr import ArTransformer
from funasr.register import tables


@tables.register("model_classes", "ArConformer")
class ArConformer(ArTransformer):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
