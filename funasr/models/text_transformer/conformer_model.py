import logging

import torch

from funasr.models.text_transformer.model import TextTransformer
from funasr.register import tables


@tables.register("model_classes", "TextConformer")
class TextConformer(TextTransformer):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
