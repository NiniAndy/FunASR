import logging

import torch

from funasr.models.accent_recognition.transformer_ar_only import TransformerOnlyAr
from funasr.register import tables


@tables.register("model_classes", "ConformerOnlyAr")
class ConformerOnlyAr(TransformerOnlyAr):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
