import logging

import torch

from funasr.models.accent_recognition.transformer_anchor_contrast_model import TransformerAnchorContrastAr
from funasr.register import tables


@tables.register("model_classes", "ConformerAnchorContrastAr")
class ConformerAnchorContrastAr(TransformerAnchorContrastAr):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
