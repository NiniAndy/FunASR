import torch
import torchaudio
import torch.nn as nn
import numpy as np
import torch.nn.functional as F




#######################  audio loader  ########################
EPS = np.finfo(float).eps

def load_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = normalized_wave(waveform)
    return waveform, sample_rate

def normalized_wave(audio, target_level=-25):
    '''根据音频的真有效值归一化'''
    audio = np.array(audio.squeeze(0))
    audio = audio / (max(abs(audio)) + EPS)
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    return torch.tensor(audio).unsqueeze(0)




#################  FBank  ###########################
class Delta(nn.Module):
    def __init__(self, order=1, window_size=2):
        super(Delta, self).__init__()

        self.order = order
        self.window_size = window_size

        filters = self._create_filters(order, window_size)
        self.register_buffer("filters", filters)
        self.padding = (0, (filters.shape[-1] - 1) // 2)

    def forward(self, x):
        x = x.unsqueeze(0)
        return F.conv2d(x, weight=self.filters, padding=self.padding)[0]

    def _create_filters(self, order, window_size):
        scales = [[1.0]]
        for i in range(1, order + 1):
            prev_offset = (len(scales[i - 1]) - 1) // 2
            curr_offset = prev_offset + window_size

            curr = [0] * (len(scales[i - 1]) + 2 * window_size)
            normalizer = 0.0
            for j in range(-window_size, window_size + 1):
                normalizer += j * j
                for k in range(-prev_offset, prev_offset + 1):
                    curr[j + k + curr_offset] += (j * scales[i - 1][k + prev_offset])
            curr = [x / normalizer for x in curr]
            scales.append(curr)

        max_len = len(scales[-1])
        for i, scale in enumerate(scales[:-1]):
            padding = (max_len - len(scale)) // 2
            scales[i] = [0] * padding + scale + [0] * padding

        return torch.tensor(scales).unsqueeze(1).unsqueeze(1)


class ExtractFbankFeature(nn.Module):
    """
        特征提取输出维度[1, 40, n_frame]
    """

    def __init__(self, config):
        super(ExtractFbankFeature, self).__init__()
        # 定义特征提取方法 输出维度[n_frame, 40]
        self.num_mel_bins = config.FEATURE.num_mel_bins
        self.delta_order = config.FEATURE.delta_order
        self.delta_window_size = config.FEATURE.delta_window_size
        self.spectrum_normalize = config.FEATURE.spectrum_normalize
        self.extract_fn = torchaudio.compliance.kaldi.fbank
        self.audio_config = dict(window_type=config.AUDIO.window,
                                 frame_length=config.AUDIO.window_size * 1000,
                                 frame_shift=config.AUDIO.window_stride * 1000)
        self.delta = Delta(self.delta_order, self.delta_window_size) if self.delta_order > 1 else None

    def forward(self, waveform, sample_rate):
        x = self.extract_fn(waveform, num_mel_bins=self.num_mel_bins, channel=-1, sample_frequency=sample_rate, **self.audio_config)
        x = x.transpose(0, 1).unsqueeze(0).detach()  # 输出维度[1, 40, n_frame]
        if self.delta:
            x = self.delta(x)
        if self.spectrum_normalize:
            x = (x - x.mean(2, keepdim=True)) / (1e-10 + x.std(2, keepdim=True))
        x = x.reshape(-1, x.size(2)).detach()
        return x  # [dim, n_frame]