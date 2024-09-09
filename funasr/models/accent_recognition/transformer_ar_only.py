import logging
from typing import Union, Dict, List, Tuple, Optional

import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import copy

from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.models.ctc.ctc import CTC
from funasr.models.transformer.utils.add_sos_eos import add_sos_eos
from funasr.metrics.compute_acc import th_accuracy

# from funasr.models.e2e_asr_common import ErrorCalculator
from funasr.train_utils.device_funcs import force_gatherable
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.utils import postprocess_utils
from funasr.utils.datadir_writer import DatadirWriter
from funasr.register import tables

from funasr.models.transformer.utils.subsampling import HubertFeatureEncoder
from funasr.models.transformer.embedding import PositionalEncoding
from funasr.models.accent_recognition.fusion_fuction import AddFusionFunction, ConcatFusionFunction

import os
import yaml
from funasr.auto.auto_model import AutoModel


@tables.register("model_classes", "TransformerOnlyAr")
class TransformerOnlyAr(nn.Module):
    """
    只有encoder的模型做ar训练
    1.加载一个训练好的base模型，用lora训练
    2.利用该lora的模型的encoder_out做ar识别
    """

    def __init__(
        self,
        specaug: str = None,
        specaug_conf: dict = None,
        normalize: str = None,
        normalize_conf: dict = None,
        encoder: str = None,
        encoder_conf: dict = None,
        ctc: str = None,
        ctc_conf: dict = None,
        interctc_weight: float = 0.0,
        input_size: int = 80,
        vocab_size: int = -1,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        # extract_feats_in_collect_stats: bool = True,
        share_embedding: bool = False,
        # preencoder: Optional[AbsPreEncoder] = None,
        # postencoder: Optional[AbsPostEncoder] = None,
        **kwargs,
    ):

        super().__init__()

        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)

        if normalize is not None:
            normalize_class = tables.normalize_classes.get(normalize)
            normalize = normalize_class(**normalize_conf)

        encoder_class = tables.encoder_classes.get(encoder)
        ar_encoder = encoder_class(input_size=input_size, **encoder_conf)
        encoder_output_size = ar_encoder.output_size()



        positional_dropout_rate = encoder_conf.get("positional_dropout_rate", 0.0)
        ctc_tokenizer = nn.Sequential(
            nn.Embedding(vocab_size, encoder_output_size),
            PositionalEncoding(encoder_output_size, positional_dropout_rate),
        )
        # 冻结ctc_tokenizer
        for param in ctc_tokenizer.parameters():
            param.requires_grad = False

        text_language_vocab_path = kwargs.get("text_language_vocab_path", None)
        assert text_language_vocab_path is not None, "text_language_vocab_path is required"
        text_language_vocab = {}
        with open(kwargs['text_language_vocab_path'], "r") as f:
            for line in f:
                line = line.strip()
                text_language_vocab[line] = len(text_language_vocab)

        with open(kwargs['text_language_vocab_path'], 'r') as file:
            dialects = [line.strip() for line in file]
        self.dial2id = {dialect: idx for idx, dialect in enumerate(dialects)}
        self.id2dial = {idx: dialect for idx, dialect in enumerate(dialects)}

        accent_size = len(text_language_vocab)

        if len(ar_encoder.interctc_layer_idx) != 0:
            x_a_size = encoder_output_size * len(ar_encoder.interctc_layer_idx)
        else:
            x_a_size = encoder_output_size
        x_hyp_size = encoder_output_size
        ar_decoder_input_size = encoder_output_size
        self.ar_decoder = AddFusionFunction(x_hyp_size, x_a_size, ar_decoder_input_size, accent_size)

        ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **ctc_conf)

        tokenizer = kwargs.get("tokenizer", None)
        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1

        if tokenizer is not None:
            token2id = tokenizer.token2id
            self.blank_id = token2id.get("<blank>", blank_id)
            self.sos = token2id.get("<s>", self.sos)
            self.eos = token2id.get("</s>", self.eos)

            if hasattr(tokenizer, 'add_special_token_list'):
                add_special_token_list = tokenizer.add_special_token_list
            else:
                add_special_token_list = False
            if add_special_token_list:
                self.start_id_of_special_tokens = len(token2id) - len(add_special_token_list)

        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.specaug = specaug
        self.normalize = normalize
        self.ar_encoder = ar_encoder
        self.tokenizer = tokenizer

        if not hasattr(self.ar_encoder, "interctc_use_conditioning"):
            self.ar_encoder.interctc_use_conditioning = False
        if self.ar_encoder.interctc_use_conditioning:
            self.ar_encoder.conditioning_layer = nn.Linear(vocab_size, self.ar_encoder.output_size())

        self.error_calculator = None
        self.ar_ctc = ctc
        self.ar_ctc_tokenizer = ctc_tokenizer

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None

        self.total_num = 0
        self.total_right = 1


    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        text_language: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size = speech.shape[0]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # 计算ctc
        loss_ctc, cer_ctc = None, None
        loss_ctc, cer_ctc, ctc_ys_hat = self._calc_ctc_loss(encoder_out, encoder_out_lens, text, text_lengths)

        # Regularization
        ctc_ys_hat = self.replace_blanks_with_nearest(ctc_ys_hat)
        x_hyp = self.ar_ctc_tokenizer(ctc_ys_hat)

        # Acoustic-semantic disentanglement
        intermediate_out_list = [out for _, out in intermediate_outs]
        x_a = torch.cat(intermediate_out_list, dim=-1)

        _, y_dal_hat, loss_dal, acc_dal = self.ar_decoder(x_hyp, x_a, text_language)

        loss = 0.3*loss_ctc + loss_dal

        stats = dict()
        # Collect CTC branch stats
        stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
        stats["cer_ctc"] = cer_ctc

        # Collect Attn branch stats
        stats["loss_att"] = loss_dal.detach() if loss_dal is not None else None
        stats["acc"] = acc_dal
        stats["cer"] = None
        stats["wer"] = None

        # Collect total loss stats
        stats["loss"] = torch.clone(loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((text_lengths + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight


    def replace_blanks_with_nearest(self, ys_hat):
        # Forward fill: replace each blank with the next non-blank token
        for i in range(ys_hat.size(1) - 2, -1, -1):
            mask = ys_hat[:, i] == 0
            ys_hat[:, i][mask] = ys_hat[:, i + 1][mask]

        # Backward fill: replace remaining blanks with the previous non-blank token
        for i in range(1, ys_hat.size(1)):
            mask = ys_hat[:, i] == 0
            ys_hat[:, i][mask] = ys_hat[:, i - 1][mask]

        return ys_hat

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                ind: int
        """
        with autocast(False):

            # 如果encoder的embed是Hubert/Wav2Vec2的embed则需要先进行cnn再做增强
            if isinstance(self.ar_encoder.embed, HubertFeatureEncoder):
                speech = speech.transpose(1, 2)
                speech, speech_lengths = self.ar_encoder.feature_extractor_forward(speech, speech_lengths)
                speech, pos_emb = speech
                speech, speech_lengths = self.specaug(speech, speech_lengths)
                speech = (speech, pos_emb)
                if self.ar_encoder.interctc_use_conditioning:
                    encoder_out, encoder_out_lens, _ = self.ar_encoder.encoder_forward(speech, speech_lengths, ctc=self.ar_ctc)
                else:
                    encoder_out, encoder_out_lens, _ = self.ar_encoder.encoder_forward(speech, speech_lengths)
            else:
                # Data augmentation
                if self.specaug is not None and self.training:
                    speech, speech_lengths = self.specaug(speech, speech_lengths)

                # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
                if self.normalize is not None:
                    speech, speech_lengths = self.normalize(speech, speech_lengths)

                # Forward ar_encoder
                # feats: (Batch, Length, Dim)
                # -> encoder_out: (Batch, Length2, Dim2)
                if self.ar_encoder.interctc_use_conditioning:
                    encoder_out, encoder_out_lens, _ = self.ar_encoder(speech, speech_lengths, ctc=self.ar_ctc)
                else:
                    encoder_out, encoder_out_lens, _ = self.ar_encoder(speech, speech_lengths)

        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens


    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        text_language: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens, text_language)

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        ys_hat = self.ar_ctc.argmax(encoder_out).data
        loss_ctc = self.ar_ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc, ys_hat





    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):




        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")


        meta_data = {}
        if (
            isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
        ):  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(
                data_in,
                fs=frontend.fs,
                audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"),
                tokenizer=tokenizer,
            )
            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(
                audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            meta_data["batch_data_time"] = (
                speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
            )

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])
        # Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # 计算ctc
        ctc_ys_hat = self.ar_ctc.argmax(encoder_out).data

        # Regularization
        ctc_ys_hat = self.replace_blanks_with_nearest(ctc_ys_hat)
        x_hyp = self.ar_ctc_tokenizer(ctc_ys_hat)

        # Acoustic-semantic disentanglement
        intermediate_out_list = [out for _, out in intermediate_outs]
        x_a = torch.cat(intermediate_out_list, dim=-1)

        results = self.ar_decoder.inference(x_hyp, x_a)

        text_language = kwargs.get("text_language", None)
        self.total_num += 1
        if text_language is not None:
            lab = self.dial2id[text_language[0]]
            if lab == results[0]:
                self.total_right += 1

        acc = self.total_right / self.total_num
        results = str(acc)
        print(f"acc: {acc}")


        return results, meta_data
