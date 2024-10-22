"""SpecAugment module."""

from typing import Optional
from typing import Sequence
from typing import Union
import random

from funasr.models.specaug.mask_along_axis import MaskAlongAxis
from funasr.models.specaug.mask_along_axis import MaskAlongAxisVariableMaxWidth
from funasr.models.specaug.mask_along_axis import MaskAlongAxisLFR
from funasr.models.specaug.time_warp import TimeWarp
from funasr.register import tables

import torch.nn as nn
import ast


@tables.register("specaug_classes", "SpecAug")
@tables.register("specaug_classes", "specaug")
class SpecAug(nn.Module):
    """Implementation of SpecAug.

    Reference:
        Daniel S. Park et al.
        "SpecAugment: A Simple Data
         Augmentation Method for Automatic Speech Recognition"

    .. warning::
        When using cuda mode, time_warp doesn't have reproducibility
        due to `torch.nn.functional.interpolate`.

    """

    def __init__(
        self,
        apply_time_warp: bool = True,
        time_warp_window: int = 5,
        time_warp_mode: str = "bicubic",
        apply_freq_mask: bool = True,
        freq_mask_width_range: Union[int, Sequence[int]] = (0, 20),
        num_freq_mask: int = 2,
        apply_time_mask: bool = True,
        time_mask_width_range: Optional[Union[int, Sequence[int]]] = None,
        time_mask_width_ratio_range: Optional[Union[float, Sequence[float]]] = None,
        num_time_mask: int = 2,
    ):
        if not apply_time_warp and not apply_time_mask and not apply_freq_mask:
            raise ValueError("Either one of time_warp, time_mask, or freq_mask should be applied")
        if (
            apply_time_mask
            and (time_mask_width_range is not None)
            and (time_mask_width_ratio_range is not None)
        ):
            raise ValueError(
                'Either one of "time_mask_width_range" or '
                '"time_mask_width_ratio_range" can be used'
            )
        super().__init__()
        self.apply_time_warp = apply_time_warp
        self.apply_freq_mask = apply_freq_mask
        self.apply_time_mask = apply_time_mask

        if apply_time_warp:
            self.time_warp = TimeWarp(window=time_warp_window, mode=time_warp_mode)
        else:
            self.time_warp = None

        if apply_freq_mask:
            self.freq_mask = MaskAlongAxis(
                dim="freq",
                mask_width_range=freq_mask_width_range,
                num_mask=num_freq_mask,
            )
        else:
            self.freq_mask = None

        if apply_time_mask:
            if time_mask_width_range is not None:
                self.time_mask = MaskAlongAxis(
                    dim="time",
                    mask_width_range=time_mask_width_range,
                    num_mask=num_time_mask,
                )
            elif time_mask_width_ratio_range is not None:
                self.time_mask = MaskAlongAxisVariableMaxWidth(
                    dim="time",
                    mask_width_ratio_range=time_mask_width_ratio_range,
                    num_mask=num_time_mask,
                )
            else:
                raise ValueError(
                    'Either one of "time_mask_width_range" or '
                    '"time_mask_width_ratio_range" should be used.'
                )
        else:
            self.time_mask = None

    def forward(self, x, x_lengths=None):
        if self.time_warp is not None:
            x, x_lengths = self.time_warp(x, x_lengths)
        if self.freq_mask is not None:
            x, x_lengths = self.freq_mask(x, x_lengths)
        if self.time_mask is not None:
            x, x_lengths = self.time_mask(x, x_lengths)
        return x, x_lengths


@tables.register("specaug_classes", "SpecAugLFR")
class SpecAugLFR(nn.Module):
    """Implementation of SpecAug.
    lfr_rate：low frame rate
    """

    def __init__(
        self,
        apply_time_warp: bool = True,
        time_warp_window: int = 5,
        time_warp_mode: str = "bicubic",
        apply_freq_mask: bool = True,
        freq_mask_width_range: Union[int, Sequence[int]] = (0, 20),
        num_freq_mask: int = 2,
        lfr_rate: int = 0,
        apply_time_mask: bool = True,
        time_mask_width_range: Optional[Union[int, Sequence[int]]] = None,
        time_mask_width_ratio_range: Optional[Union[float, Sequence[float]]] = None,
        num_time_mask: int = 2,
    ):
        if not apply_time_warp and not apply_time_mask and not apply_freq_mask:
            raise ValueError("Either one of time_warp, time_mask, or freq_mask should be applied")
        if (
            apply_time_mask
            and (time_mask_width_range is not None)
            and (time_mask_width_ratio_range is not None)
        ):
            raise ValueError(
                'Either one of "time_mask_width_range" or '
                '"time_mask_width_ratio_range" can be used'
            )
        super().__init__()
        self.apply_time_warp = apply_time_warp
        self.apply_freq_mask = apply_freq_mask
        self.apply_time_mask = apply_time_mask

        if apply_time_warp:
            self.time_warp = TimeWarp(window=time_warp_window, mode=time_warp_mode)
        else:
            self.time_warp = None

        if apply_freq_mask:
            self.freq_mask = MaskAlongAxisLFR(
                dim="freq",
                mask_width_range=freq_mask_width_range,
                num_mask=num_freq_mask,
                lfr_rate=lfr_rate + 1,
            )

        else:
            self.freq_mask = None

        if apply_time_mask:
            if time_mask_width_range is not None:
                self.time_mask = MaskAlongAxisLFR(
                    dim="time",
                    mask_width_range=time_mask_width_range,
                    num_mask=num_time_mask,
                    lfr_rate=lfr_rate + 1,
                )
            elif time_mask_width_ratio_range is not None:
                self.time_mask = MaskAlongAxisVariableMaxWidth(
                    dim="time",
                    mask_width_ratio_range=time_mask_width_ratio_range,
                    num_mask=num_time_mask,
                )
            else:
                raise ValueError(
                    'Either one of "time_mask_width_range" or '
                    '"time_mask_width_ratio_range" should be used.'
                )
        else:
            self.time_mask = None

    def forward(self, x, x_lengths=None):
        if self.time_warp is not None:
            x, x_lengths = self.time_warp(x, x_lengths)
        if self.freq_mask is not None:
            x, x_lengths = self.freq_mask(x, x_lengths)
        if self.time_mask is not None:
            x, x_lengths = self.time_mask(x, x_lengths)
        return x, x_lengths



@tables.register("specaug_classes", "SpecAugWav2Vec2")
class SpecAugWav2Vec2(nn.Module):
    """和wav2vec2相似的谱增强方法"""

    def __init__(
            self,
            use_spec_augment: bool = False,
            spec_augment_prob: float = 0.3,
            apply_time_warp: bool = True,
            time_warp_window: int = 80,
            time_warp_mode: str = "bicubic",
            apply_freq_mask: bool = True,
            freq_mask_width_range: Union[int, Sequence[int]] = (0, 100),
            num_freq_mask: int = 2,
            apply_time_mask: bool = True,
            time_mask_width_range: Optional[Union[int, Sequence[int]]] = (0, 5),
            time_mask_width_ratio_range: Optional[Union[float, Sequence[float]]] = (0, 0.1),
            num_time_mask: int = 2,
    ):
        super(SpecAugWav2Vec2, self).__init__()
        self.spec_augment_prob = spec_augment_prob

        if freq_mask_width_range is not None:
            freq_mask_width_range = ast.literal_eval(freq_mask_width_range)
        if time_mask_width_range is not None:
            time_mask_width_range = ast.literal_eval(time_mask_width_range)
        if time_mask_width_ratio_range is not None:
            time_mask_width_ratio_range = ast.literal_eval(time_mask_width_ratio_range)

        if use_spec_augment:
            from funasr.models.pretrain_asr.spec_augment import wav2vec2_SpecAug
            self.spec_augment = wav2vec2_SpecAug(
                apply_time_warp,
                time_warp_window,
                time_warp_mode,
                apply_freq_mask,
                freq_mask_width_range,
                num_freq_mask,
                apply_time_mask,
                time_mask_width_range,
                time_mask_width_ratio_range,
                num_time_mask,
            )
        else:
            self.spec_augment = None

    def forward(self, x, x_lengths=None):
        if random.random() < self.spec_augment_prob and self.spec_augment is not None:
            x, x_lengths = self.spec_augment(x, x_lengths)
        return x, x_lengths




