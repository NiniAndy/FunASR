import logging

import torch

from funasr.models.text_transformer_2.model import TextTransformer2
from funasr.register import tables


@tables.register("model_classes", "TextConformer2")
class TextConformer2(TextTransformer2):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
