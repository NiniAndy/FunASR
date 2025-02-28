import torch
import random
from itertools import chain

from pypinyin import lazy_pinyin, INITIALS, FINALS_TONE3

from funasr.register import tables
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video

def make_pny(sentence):
    origin_initials = lazy_pinyin(sentence, INITIALS)  # 声母
    origin_vowel = lazy_pinyin(sentence, FINALS_TONE3)  # 韵母和音调
    origin_pinyin = [origin_initials[j] + origin_vowel[j] for j in range(len(origin_initials))]
    return origin_pinyin


@tables.register("dataset_classes", "AudioPNYDataset")
class AudioPNYDataset(torch.utils.data.Dataset):
    """
    AudioPNYDataset
    """

    def __init__(
        self,
        path,
        index_ds: str = None,
        frontend=None,
        tokenizer=None,
        pny_tokenizer=None,
        is_training: bool = True,
        int_pad_value: int = -1,
        float_pad_value: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)

        self.preprocessor_speech = None
        self.preprocessor_text = None
        self.preprocessor_err = None

        if is_training:
            preprocessor_speech = kwargs.get("preprocessor_speech", None)
            if preprocessor_speech:
                preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
                preprocessor_speech = preprocessor_speech_class(**kwargs.get("preprocessor_speech_conf"))
            self.preprocessor_speech = preprocessor_speech

            preprocessor_text = kwargs.get("preprocessor_text", None)
            if preprocessor_text:
                preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
                preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
            self.preprocessor_text = preprocessor_text

            preprocessor_err = kwargs.get("preprocessor_err", None)
            if preprocessor_err:
                preprocessor_err_class = tables.preprocessor_classes.get(preprocessor_err)
                preprocessor_err = preprocessor_err_class(**kwargs.get("preprocessor_err_conf"))
            self.preprocessor_err = preprocessor_err



        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer
        self.pny_tokenizer = pny_tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        item = self.index_ds[index]
        # import pdb;
        # pdb.set_trace()
        source = item["source"]
        data_src = load_audio_text_image_video(source, fs=self.fs)
        if self.preprocessor_speech:
            data_src = self.preprocessor_speech(data_src, fs=self.fs)

        speech, speech_lengths = extract_fbank(
            data_src,
            data_type=self.data_type,
            frontend=self.frontend,
            is_final=True
        )  # speech: [b, T, d]

        target = item["target"]
        if self.preprocessor_text:
            target = self.preprocessor_text(target)

        if self.tokenizer:
            ids = self.tokenizer.encode(target)
            text = torch.tensor(ids, dtype=torch.int64)
        else:
            ids = target
            text = ids
        ids_lengths = len(ids)
        text_lengths = torch.tensor([ids_lengths], dtype=torch.int32)

        if self.preprocessor_err:
            wrong_text, _ = self.preprocessor_err(target)
        else:
            wrong_text = target


        pny_list = make_pny(wrong_text)
        # blank = ['<blank>']
        # pny_list = list(chain(*zip(pny_list, blank * (len(pny_list) - 1)), [pny_list[-1]]))
        pny = self.pny_tokenizer.tokens2ids(pny_list)
        pny = torch.tensor(pny, dtype=torch.int64)
        pny_lengths = torch.tensor([pny.size(0)], dtype=torch.int32)

        # wrong_pny = torch.tensor(wrong_pny, dtype=torch.int64)
        # pny_lengths = torch.tensor([wrong_pny.size(0)], dtype=torch.int32)
        # text_lengths = torch.tensor([text.size(0)], dtype=torch.int32)

        return {
            "speech": speech[0, :, :],
            "speech_lengths": speech_lengths,
            "text": text,
            "text_lengths": text_lengths,
            "pny": pny,
            "pny_lengths": pny_lengths
        }

    def collator(self, samples: list = None):
        outputs = {}
        for sample in samples:
            for key in sample.keys():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(sample[key])

        for key, data_list in outputs.items():
            if isinstance(data_list[0], torch.Tensor):
                if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                    pad_value = self.int_pad_value
                else:
                    pad_value = self.float_pad_value

                outputs[key] = torch.nn.utils.rnn.pad_sequence(
                    data_list, batch_first=True, padding_value=pad_value
                )
        return outputs

