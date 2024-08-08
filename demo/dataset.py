import json
import os
import random
import re

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def collate_fn(batch):
    assert isinstance(batch, list)
    feats_length = torch.tensor([x['wav'].size(1) for x in batch], dtype=torch.int32)
    order = torch.argsort(feats_length, descending=True)

    feats_lengths = torch.tensor([batch[i]['wav'].size(1) for i in order], dtype=torch.int32)
    sorted_feats = [batch[i]['wav'].squeeze(0) for i in order]
    sorted_keys = [batch[i]['key'] for i in order]
    sorted_txt = [torch.tensor(batch[i]['label'], dtype=torch.int64) for i in order]
    sorted_acc = [torch.tensor(batch[i]['acc_label'], dtype=torch.int64).unsqueeze(-1) for i in order]
    sorted_km = [torch.tensor(batch[i]['km'], dtype=torch.int64) for i in order]
    sorted_pny = [torch.tensor(batch[i]['pny_label'], dtype=torch.int64) for i in order]

    km_lengths = torch.tensor([x.size(0) for x in sorted_km], dtype=torch.int32)
    pny_lengths = torch.tensor([x.size(0) for x in sorted_pny], dtype=torch.int32)
    txt_lengths = torch.tensor([x.size(0) for x in sorted_txt], dtype=torch.int32)

    padded_feats = pad_sequence(sorted_feats, batch_first=True, padding_value=0).unsqueeze(1)
    padding_txt = pad_sequence(sorted_txt, batch_first=True, padding_value=-1)
    padding_km = pad_sequence(sorted_km, batch_first=True, padding_value=17)
    padding_pny = pad_sequence(sorted_pny, batch_first=True, padding_value=-1)
    padding_acc = pad_sequence(sorted_acc, batch_first=True, padding_value=-1).squeeze(-1)

    return (sorted_keys, padded_feats, padding_txt, feats_lengths, txt_lengths, [padding_km, km_lengths, padding_acc])



class SpectrogramDataset(Dataset):
    def __init__(
            self,
            dataset_conf,
            file_list,
            symbol_table,
            pny_table,
            acc_table,
            bpe_model=None,
            non_lang_syms=None, ):
        super(SpectrogramDataset, self).__init__()

        self.symbol_table = symbol_table
        self.pny_table = pny_table
        self.acc_table = acc_table
        self.bpe_model = bpe_model
        self.non_lang_syms = non_lang_syms
        self.dataset_conf = dataset_conf

        self.filter_conf = dataset_conf.get('filter_conf', {})
        self.resample_conf = dataset_conf.get('resample_conf', {})
        self.speed_perturb_flag = dataset_conf.get('speed_perturb', False)
        self.feats_type = dataset_conf.get('feats_type', 'fbank')
        if self.feats_type == 'fbank':
            self.fbank_conf = dataset_conf.get('fbank_conf', {})
        if self.feats_type == 'mfcc':
            self.mfcc_conf = dataset_conf.get('mfcc_conf', {})
        self.spec_aug_flag = dataset_conf.get('spec_aug', True)
        if self.spec_aug_flag:
            self.spec_aug_conf = dataset_conf.get('spec_aug_conf', {})
        self.spec_sub_flag = dataset_conf.get('spec_sub', False)
        if self.spec_sub_flag:
            self.spec_sub_conf = dataset_conf.get('spec_sub_conf', {})
        self.spec_trim_flag = dataset_conf.get('spec_trim', False)
        if self.spec_trim_flag:
            self.spec_trim_conf = dataset_conf.get('spec_trim_conf', {})

        self.split_with_space = dataset_conf.get('split_with_space', False)

        self.file_list = file_list

    def __getitem__(self, index):
        sample = {}
        item = self.file_list[index]
        obj = json.loads(item)
        sample['key'] = obj['key']
        sample['txt'] = obj['txt']
        sample['wav'] = os.path.join("/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/", obj['wav'])
        sample['acc'] = obj['acc']
        sample['pny'] = obj['pny']
        sample['km'] = [int(x) for x in obj['km'].split()]
        # {''key', 'wav', 'txt', 'acc', 'pny'}

        sample = self.parse_raw(sample)
        # {'sample_rate'}

        # sample = self.resample(sample, **self.resample_conf)
        # if self.speed_perturb_flag:
        #     sample = self.speed_perturb(sample)

        # if self.feats_type == 'fbank':
        #     sample = self.compute_fbank(sample, **self.fbank_conf)
        #     # {'feat'}
        # elif self.feats_type == 'mfcc':
        #     sample = self.compute_mfcc(sample, **self.mfcc_conf)
        #     # {'feat'}

        # if self.spec_aug_flag:
        #     sample = self.spec_aug(sample, **self.spec_aug_conf)
        # if self.spec_sub_flag:
        #     sample = self.spec_sub(sample, **self.spec_sub_conf)
        # if self.spec_trim_flag:
        #     sample = self.spec_trim(sample, **self.spec_trim_conf)

        sample = self.tokenize(sample, self.split_with_space)
        # {'tokens', 'label'}
        sample = self.pny_tokenize(sample)
        # # {'pny_label'}
        sample = self.acc_tokenize(sample)
        # {'acc_label'}

        return {
            'key': sample['key'],
            'wav': sample['wav'],
            'acc_label': sample['acc_label'],
            'label': sample['label'],
            'pny_label': sample['pny_label'],
            'km': sample['km'],
            }

    def __len__(self):
        return len(self.file_list)

    def parse_raw(self, sample):
        wav_file = sample['wav']
        if 'start' in sample:
            assert 'end' in sample
            sample_rate = torchaudio.backend.sox_io_backend.info(wav_file).sample_rate
            start_frame = int(sample['start'] * sample_rate)
            end_frame = int(sample['end'] * sample_rate)
            waveform, _ = torchaudio.backend.sox_io_backend.load(
                filepath=wav_file,
                num_frames=end_frame - start_frame,
                frame_offset=start_frame)
        else:
            waveform, sample_rate = torchaudio.load(wav_file)
        sample['wav'] = waveform
        sample['sample_rate'] = sample_rate
        return sample

    def resample(self, sample, resample_rate=16000):
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        if sample_rate != resample_rate:
            sample['sample_rate'] = resample_rate
            sample['wav'] = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        return sample

    def speed_perturb(self, sample, speeds=None):
        if speeds is None:
            speeds = [0.9, 1.0, 1.1]
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        speed = random.choice(speeds)
        if speed != 1.0:
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform,
                sample_rate,
                [['speed', str(speed)], ['rate', str(sample_rate)]])
        sample['wav'] = wav
        return sample

    ###############################################################

    ###############################################################
    # 计算谱
    # Return: +{'feat': mat}
    def compute_fbank(
            self,
            sample,
            num_mel_bins=23,
            frame_length=25,
            frame_shift=10,
            dither=0.0):

        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=dither,
                          energy_floor=0.0,
                          sample_frequency=sample_rate)
        sample['feat'] = mat
        return sample

    def compute_mfcc(self,
                     sample,
                     num_mel_bins=23,
                     frame_length=25,
                     frame_shift=10,
                     dither=0.0,
                     num_ceps=40,
                     high_freq=0.0,
                     low_freq=20.0):

        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.mfcc(waveform,
                         num_mel_bins=num_mel_bins,
                         frame_length=frame_length,
                         frame_shift=frame_shift,
                         dither=dither,
                         num_ceps=num_ceps,
                         high_freq=high_freq,
                         low_freq=low_freq,
                         sample_frequency=sample_rate)
        sample['feat'] = mat
        return sample

    def spec_aug(self,
                 sample,
                 num_t_mask=2,
                 num_f_mask=2,
                 max_t=50,
                 max_f=10,
                 max_w=80):
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        max_freq = y.size(1)
        # time mask
        for i in range(num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask
        for i in range(num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        sample['feat'] = y
        return sample

    def spec_sub(sample, max_t=20, num_t_sub=3):
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        for i in range(num_t_sub):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            # only substitute the earlier time chosen randomly for current time
            pos = random.randint(0, start)
            y[start:end, :] = x[start - pos:end - pos, :]
        sample['feat'] = y
        return sample

    def spec_trim(self, sample, max_t=20):
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        max_frames = x.size(0)
        length = random.randint(1, max_t)
        if length < max_frames / 2:
            y = x.clone().detach()[:max_frames - length]
            sample['feat'] = y
        return sample

    ###############################################################

    ###############################################################
    # Tokenize
    def __tokenize_by_bpe_model(self, sp, txt):
        tokens = []
        # CJK(China Japan Korea) unicode range is [U+4E00, U+9FFF], ref:
        # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        pattern = re.compile(r'([\u4e00-\u9fff])')
        # Example:
        #   txt   = "你好 ITS'S OKAY 的"
        #   chars = ["你", "好", " ITS'S OKAY ", "的"]
        chars = pattern.split(txt.upper())
        mix_chars = [w for w in chars if len(w.strip()) > 0]
        for ch_or_w in mix_chars:
            # ch_or_w is a single CJK charater(i.e., "你"), do nothing.
            if pattern.fullmatch(ch_or_w) is not None:
                tokens.append(ch_or_w)
            # ch_or_w contains non-CJK charaters(i.e., " IT'S OKAY "),
            # encode ch_or_w using bpe_model.
            else:
                for p in sp.encode_as_pieces(ch_or_w):
                    tokens.append(p)
        return tokens

    def tokenize(self,
                 sample,
                 split_with_space=False):
        if self.non_lang_syms is not None:
            non_lang_syms_pattern = re.compile(r"(\[[^\[\]]+\]|<[^<>]+>|{[^{}]+})")
        else:
            self.non_lang_syms = {}
            non_lang_syms_pattern = None

        if self.bpe_model is not None:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.load(self.bpe_model)
        else:
            sp = None

        txt = sample['txt'].strip()
        if non_lang_syms_pattern is not None:
            parts = non_lang_syms_pattern.split(txt.upper())
            parts = [w for w in parts if len(w.strip()) > 0]
        else:
            parts = [txt]

        label = []
        tokens = []
        for part in parts:
            if part in self.non_lang_syms:
                tokens.append(part)
            else:
                if self.bpe_model is not None:
                    tokens.extend(self.__tokenize_by_bpe_model(sp, part))
                else:
                    if split_with_space:
                        part = part.split(" ")
                    for ch in part:
                        if ch == ' ':
                            ch = "▁"
                        tokens.append(ch)

            for ch in tokens:
                if ch in self.symbol_table:
                    label.append(self.symbol_table[ch])
                elif '<unk>' in self.symbol_table:
                    label.append(self.symbol_table['<unk>'])

            sample['tokens'] = tokens
            sample['label'] = label
            return sample

    def pny_tokenize(self, sample):
        pny = sample['pny'].strip()
        parts = pny.split(" ")
        pny_label = []
        for ch in parts:
            if ch in self.pny_table:
                pny_label.append(self.pny_table[ch])
            elif '<unk>' in self.pny_table:
                pny_label.append(self.pny_table['<unk>'])
        sample['pny_label'] = pny_label
        return sample

    def acc_tokenize(self, sample):
        acc = sample['acc'].strip()
        acc_label = self.acc_table[acc]
        sample['acc_label'] = acc_label
        return sample
