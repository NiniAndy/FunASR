import logging

import torch

from funasr.models.accent_recognition.transformer_lora_asrNar import MultiEmbTransformerAsrNAr
from funasr.register import tables


@tables.register("model_classes", "MultiEmbConformerAsrNAr")
class MultiEmbConformerAsrNAr(MultiEmbTransformerAsrNAr):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
