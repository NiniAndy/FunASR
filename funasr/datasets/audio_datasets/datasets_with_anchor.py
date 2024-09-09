import torch
import random

from funasr.register import tables
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video



@tables.register("dataset_classes", "AudioWithDialectDatasetAndAnchor")
class AudioWithDialectDatasetAndAnchor(torch.utils.data.Dataset):
    """
    AudioDataset adding dialect
    """

    def __init__(
        self,
        path,
        index_ds: str = None,
        frontend=None,
        tokenizer=None,
        int_pad_value: int = -1,
        float_pad_value: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        # 是否使用音频加速 eg.
        # SpeechPreprocessSpeedPerturb
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf")
            )
        self.preprocessor_speech = preprocessor_speech
        # 是否使用文本预处理 eg.
        # TextPreprocessRemovePunctuation
        # TextPreprocessSegDict
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        # 音频前端 eg.
        # WavFrontend --> Fbank\MelSpectrogram
        # WavFrontendOnline --> Fbank\MelSpectrogram
        # WhisperFrontend --> whisper feature
        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.spurious_label_pad_value = kwargs.get("spurious_label_pad_value", 17)

        if hasattr(self.index_ds, 'text_language_flag') and callable(getattr(self.index_ds, 'text_language_flag')):
            self.text_language_flag = self.index_ds.text_language_flag()
            self.text_language_vocab_path = kwargs.get("text_language_vocab_path", None)
            if self.text_language_flag and self.text_language_vocab_path is None:
                raise ValueError("If text language is defined than text language vocab path have to be given")
            self.text_language_vocab = {}
            with open(self.text_language_vocab_path, "r") as f:
                for line in f:
                    line = line.strip()
                    self.text_language_vocab[line] = len(self.text_language_vocab)
        else:
            self.text_language_flag = False

        if hasattr(self.index_ds, 'audio_language_flag') and callable(getattr(self.index_ds, 'audio_language_flag')):
            self.audio_language_flag = self.index_ds.audio_language_flag()
        else:
            self.audio_language_flag = False

        if hasattr(self.index_ds, 'emo_target_flag') and callable(getattr(self.index_ds, 'emo_target_flag')):
            self.emo_target_flag = self.index_ds.emo_target_flag()
        else:
            self.emo_target_flag = False

        if hasattr(self.index_ds, 'event_target_flag') and callable(getattr(self.index_ds, 'event_target_flag')):
            self.event_target_flag = self.index_ds.event_target_flag()
        else:
            self.event_target_flag = False

        if hasattr(self.index_ds, 'with_or_wo_itn_flag') and callable(getattr(self.index_ds, 'with_or_wo_itn_flag')):
            self.with_or_wo_itn_flag = self.index_ds.with_or_wo_itn_flag()
        else:
            self.with_or_wo_itn_flag = False

        if hasattr(self.index_ds, 'spurious_label_flag') and callable(getattr(self.index_ds, 'spurious_label_flag')):
            self.spurious_label_flag = self.index_ds.spurious_label_flag()
        else:
            self.spurious_label_flag = False


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
        key = item.get("key", None)
        source = item["source"]
        data_src = load_audio_text_image_video(source, fs=self.fs)

        if self.preprocessor_speech:
            data_src = self.preprocessor_speech(data_src, fs=self.fs)
        speech, speech_lengths = extract_fbank(
            data_src, data_type=self.data_type, frontend=self.frontend, is_final=True
        )  # speech: [b, T, d]

        speech_sample = speech[0].unsqueeze(0)
        anchor_sample = speech[1].unsqueeze(0)


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

        # 格式化输出条目
        entry = {
            "key": key,
            "speech": speech_sample[0, :, :],
            "anchor": anchor_sample[0, :, :],
            "speech_lengths": speech_lengths,
            "text": text,
            "text_lengths": text_lengths,
        }

        # TODO: 增加其他条目
        # source_len = torch.tensor([item["source_len"]], dtype=torch.int32)
        # entry.update({"source_len": source_len})
        if self.text_language_flag:
            text_language = item["text_language"]
            if text_language is None:
                text_language = -1
            else:
                text_language = self.text_language_vocab[text_language]
            text_language = torch.tensor([text_language], dtype=torch.int32)
            entry.update({"text_language": text_language})
        if self.spurious_label_flag:
            spurious_label = item["spurious_label"]
            spurious_label = torch.tensor([int(x) for x in spurious_label.split()], dtype=torch.int32)
            spurious_label_lengths = torch.tensor([len(spurious_label)], dtype=torch.int32)
            entry.update({"spurious_label": spurious_label, "spurious_label_lengths": spurious_label_lengths})

        return entry




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

                if key == "spurious_label":
                    pad_value = self.spurious_label_pad_value

                outputs[key] = torch.nn.utils.rnn.pad_sequence(
                    data_list, batch_first=True, padding_value=pad_value
                )
        return outputs