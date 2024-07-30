import logging

import torch

from funasr.models.accent_recognition.transformer_model import TransformerAr
from funasr.register import tables


@tables.register("model_classes", "ConformerAr")
class Conformer(TransformerAr):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
