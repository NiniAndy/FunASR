import logging
from typing import Union, Dict, List, Tuple, Optional

import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

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

from funasr.datasets.audio_datasets.pny_datasets import make_pny
import torch.nn.functional as F

from itertools import chain

from funasr.models.text_transformer.search_method import ctc_greedy_search, attention_beam_search, mix_beam_search
from funasr.models.transformer.utils.nets_utils import make_pad_mask


@tables.register("model_classes", "TextTransformer")
class TextTransformer(nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        specaug: str = None,
        specaug_conf: dict = None,
        normalize: str = None,
        normalize_conf: dict = None,
        encoder: str = None,
        encoder_conf: dict = None,
        text_encoder: str = None,
        text_encoder_conf: dict = None,
        decoder: str = None,
        decoder_conf: dict = None,
        text_decoder: str = None,
        text_decoder_conf: dict = None,
        ctc: str = None,
        ctc_conf: dict = None,
        ctc_weight: float = 0.5,
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
        self.specaug = specaug

        if normalize is not None:
            normalize_class = tables.normalize_classes.get(normalize)
            normalize = normalize_class(**normalize_conf)
        self.normalize = normalize

        # audio encoder
        encoder_class = tables.encoder_classes.get(encoder)
        encoder = encoder_class(input_size=input_size, **encoder_conf)
        encoder_output_size = encoder.output_size()
        self.encoder = encoder

        # pny text encoder
        input_vocal_size = kwargs['pny_vocab_size']
        text_encoder_class = tables.encoder_classes.get(text_encoder)
        text_encoder = text_encoder_class(input_vocal_size=input_vocal_size, **text_encoder_conf)
        text_encoder_output_size = text_encoder.output_size()
        self.text_encoder = text_encoder

        # pny text decoder & ctc
        assert ctc_weight <= 1.0 and ctc_weight >= 0.0, "ctc_weight must be in [0.0, 1.0]"
        if ctc_weight == 1.0:
            text_decoder = None
            text_ctc = CTC(odim=vocab_size, encoder_output_size=text_encoder_output_size, **ctc_conf)
        else:
            text_decoder_class = tables.decoder_classes.get(text_decoder)
            text_decoder = text_decoder_class(
                vocab_size=vocab_size,
                encoder_output_size=text_encoder_output_size,
                **text_decoder_conf
            )
            if ctc_weight > 0.0:
                text_ctc = CTC(odim=vocab_size, encoder_output_size=text_encoder_output_size, **ctc_conf)
            else:
                text_ctc = None

        self.text_decoder = text_decoder
        self.text_ctc = text_ctc
        # self.text_ctc = None


        # audio decoder & ctc
        if ctc_weight == 1.0:
            ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **ctc_conf)
            decoder = None
        else:
            decoder_class = tables.decoder_classes.get(decoder)
            decoder = decoder_class(
                vocab_size=vocab_size,
                encoder_output_size=encoder_output_size,
                **decoder_conf
            )
            if ctc_weight > 0.0:
                ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **ctc_conf)
            else:
                ctc = None

        self.decoder = decoder
        self.ctc = ctc

        self.tokenizer = kwargs.get("tokenizer")
        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.pny_tokenizer = kwargs.get("pny_tokenizer")


        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(vocab_size, self.encoder.output_size())
        self.interctc_weight = interctc_weight


        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        self.error_calculator = None

        self.share_embedding = share_embedding
        if self.share_embedding:
            self.decoder.embed = None

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        # pny: torch.Tensor,
        # pny_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
                pny: (Batch, Length)
                pny_lengths: (Batch,)
        """
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size = speech.shape[0]
        stats = dict()

        # 1. Audio Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        pny, pny_lengths = self.ctc_out(encoder_out, encoder_out_lens)
        pny = pny.masked_fill(pny == -1, self.eos)
        if len(pny_lengths.size()) > 1:
            pny_lengths = pny_lengths[:, 0]

        # 2. Text Encoder
        text_encoder_out, text_encoder_out_lens = self.text_encode(pny, pny_lengths)

        # 3. CTC loss definition
        loss_ctc, cer_ctc, loss_text_ctc, cer_text_ctc = None, None, None, None
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(encoder_out, encoder_out_lens, text, text_lengths)
            loss_text_ctc, cer_text_ctc = self._calc_text_ctc_loss(text_encoder_out, text_encoder_out_lens, text, text_lengths)
            loss_ctc = loss_ctc + loss_text_ctc
            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["loss_text_ctc"] = loss_text_ctc.detach() if loss_text_ctc is not None else None
            stats["cer_text_ctc"] = cer_text_ctc.detach() if cer_text_ctc is not None else None
            stats["cer_ctc"] = cer_ctc


        # decoder: Attention decoder branch
        loss_text_att, acc_text_att, cer_text_att, wer_text_att, text_decoder_output_list = self._calc_text_att_loss(
            text_encoder_out,
            text_encoder_out_lens,
            text,
            text_lengths)


        stats["loss_text_att"] = loss_text_att.detach() if loss_text_att is not None else None
        stats["acc_text_att"] = acc_text_att
        stats["cer_text_att"] = cer_text_att
        stats["wer_text_att"] = wer_text_att


        loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
            encoder_out,
            encoder_out_lens,
            text,
            text_lengths,
            text_decoder_output_list)

        # 3. CTC-Att loss definition
        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * (loss_att + loss_text_att)
            # loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att

        # Collect total loss stats
        stats["loss"] = torch.clone(loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((text_lengths + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

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

            # Data augmentation
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)

            # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                speech, speech_lengths = self.normalize(speech, speech_lengths)

        # Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder(speech, speech_lengths, ctc=self.ctc)
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens


    def text_encode(
        self,
        pny: torch.Tensor,
        pny_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        Args:
                pny: (Batch, Length)
                pny_lengths: (Batch, )
                ind: int
        """
        encoder_out, encoder_out_lens, _ = self.text_encoder(pny, pny_lengths)
        return encoder_out, encoder_out_lens


    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        text_decoder_output_list: List[torch.Tensor] = None,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens, text_decoder_output_list)

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


    def _calc_text_att_loss(
        self,
        text_encoder_out: torch.Tensor,
        text_encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _, decoder_out_list = self.text_decoder.forward_by_layer(text_encoder_out, text_encoder_out_lens, ys_in_pad, ys_in_lens)

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

        return loss_att, acc_att, cer_att, wer_att, decoder_out_list

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc


    def _calc_text_ctc_loss(
        self,
        text_encoder_out: torch.Tensor,
        text_encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_text_ctc = self.text_ctc(text_encoder_out, text_encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_text_ctc = None
        if not self.training and self.error_calculator is not None:
            text_ys_hat = self.text_ctc.argmax(text_encoder_out).data
            cer_text_ctc = self.error_calculator(text_ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_text_ctc, cer_text_ctc


    def ctc_out(self, encoder_out, encoder_lens, key=None):
        with torch.no_grad():

            # encoder_out 形状为 (B, T, D)
            # 获取类别预测
            device = encoder_out.device
            ctc_probs = self.ctc.ctc_logprobs(encoder_out, self.blank_id, blank_penalty=2)
            results = ctc_greedy_search(ctc_probs, encoder_lens, self.blank_id)
            pny_lengths = []
            pnys = []
            for i in range(len(results)):
                han_id = results[i].tokens
                han_token =  "".join(self.tokenizer.ids2tokens(han_id))
                pny_id = self.han2pny(han_token)
                pny = torch.tensor(pny_id, dtype=torch.int64)
                pny_length = torch.tensor([pny.size(0)], dtype=torch.int32)

                pnys.append(pny)
                pny_lengths.append(pny_length)

            pnys = torch.nn.utils.rnn.pad_sequence(pnys, batch_first=True, padding_value=-1).to(device)
            pny_lengths = torch.nn.utils.rnn.pad_sequence(pny_lengths, batch_first=True, padding_value=-1).to(device)

        return pnys, pny_lengths


    def han2pny(self, sentence):
        if sentence == "":
            pny_token_with_blank = ["<blank>", "<blank>", "<blank>", "<blank>", "<blank>"]
        else:
            pny_token = make_pny(sentence)
            blank = ['<blank>']
            pny_token_with_blank = list(chain(*zip(pny_token, blank * (len(pny_token) - 1)), [pny_token[-1]]))
        pny_id = self.pny_tokenizer.tokens2ids(pny_token_with_blank)
        return pny_id


    def init_beam_search(
        self,
        **kwargs,
    ):
        from funasr.models.transformer.search import BeamSearch
        from funasr.models.transformer.scorers.ctc import CTCPrefixScorer
        from funasr.models.transformer.scorers.length_bonus import LengthBonus

        # 1. Build ASR model
        scorers = {}

        if self.ctc != None:
            ctc = CTCPrefixScorer(ctc=self.ctc, eos=self.eos)
            scorers.update(ctc=ctc)
        token_list = kwargs.get("token_list")
        scorers.update(
            decoder=self.decoder,
            length_bonus=LengthBonus(len(token_list)),
        )

        # 3. Build ngram model
        # ngram is not supported now
        ngram = None
        scorers["ngram"] = ngram

        weights = dict(
            decoder=1.0 - kwargs.get("decoding_ctc_weight", 0.5),
            ctc=kwargs.get("decoding_ctc_weight", 0.5),
            lm=kwargs.get("lm_weight", 0.0),
            ngram=kwargs.get("ngram_weight", 0.0),
            length_bonus=kwargs.get("penalty", 0.0),
        )
        beam_search = BeamSearch(
            beam_size=kwargs.get("beam_size", 10),
            weights=weights,
            scorers=scorers,
            sos=self.sos,
            eos=self.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if self.ctc_weight == 1.0 else "full",
        )

        self.beam_search = beam_search

    '''funasr style inference'''
    # def inference(
    #     self,
    #     data_in,
    #     data_lengths=None,
    #     key: list = None,
    #     tokenizer=None,
    #     frontend=None,
    #     **kwargs,
    # ):
    #
    #     if kwargs.get("batch_size", 1) > 1:
    #         raise NotImplementedError("batch decoding is not implemented")
    #
    #     # init beamsearch
    #     if self.beam_search is None:
    #         logging.info("enable beam_search")
    #         self.init_beam_search(**kwargs)
    #         self.nbest = kwargs.get("nbest", 1)
    #
    #     meta_data = {}
    #     if (
    #         isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
    #     ):  # fbank
    #         speech, speech_lengths = data_in, data_lengths
    #         if len(speech.shape) < 3:
    #             speech = speech[None, :, :]
    #         if speech_lengths is None:
    #             speech_lengths = speech.shape[1]
    #     else:
    #         # extract fbank feats
    #         time1 = time.perf_counter()
    #         audio_sample_list = load_audio_text_image_video(
    #             data_in,
    #             fs=frontend.fs,
    #             audio_fs=kwargs.get("fs", 16000),
    #             data_type=kwargs.get("data_type", "sound"),
    #             tokenizer=tokenizer,
    #         )
    #         time2 = time.perf_counter()
    #         meta_data["load_data"] = f"{time2 - time1:0.3f}"
    #         speech, speech_lengths = extract_fbank(
    #             audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
    #         )
    #         time3 = time.perf_counter()
    #         meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
    #         meta_data["batch_data_time"] = (
    #             speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
    #         )
    #
    #     speech = speech.to(device=kwargs["device"])
    #     speech_lengths = speech_lengths.to(device=kwargs["device"])
    #     # Encoder
    #     encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
    #     if isinstance(encoder_out, tuple):
    #         encoder_out = encoder_out[0]
    #
    #     # c. Passed the encoder result and the beam search
    #     nbest_hyps = self.beam_search(
    #         x=encoder_out[0],
    #         maxlenratio=kwargs.get("maxlenratio", 0.0),
    #         minlenratio=kwargs.get("minlenratio", 0.0),
    #     )
    #
    #     nbest_hyps = nbest_hyps[: self.nbest]
    #
    #     results = []
    #     b, n, d = encoder_out.size()
    #     for i in range(b):
    #
    #         for nbest_idx, hyp in enumerate(nbest_hyps):
    #             ibest_writer = None
    #             if kwargs.get("output_dir") is not None:
    #                 if not hasattr(self, "writer"):
    #                     self.writer = DatadirWriter(kwargs.get("output_dir"))
    #                 ibest_writer = self.writer[f"{nbest_idx + 1}best_recog"]
    #
    #             # remove sos/eos and get results
    #             last_pos = -1
    #             if isinstance(hyp.yseq, list):
    #                 token_int = hyp.yseq[1:last_pos]
    #             else:
    #                 token_int = hyp.yseq[1:last_pos].tolist()
    #
    #             # remove blank symbol id, which is assumed to be 0
    #             token_int = list(
    #                 filter(
    #                     lambda x: x != self.eos and x != self.sos and x != self.blank_id, token_int
    #                 )
    #             )
    #
    #             # Change integer-ids to tokens
    #             token = tokenizer.ids2tokens(token_int)
    #             text = tokenizer.tokens2text(token)
    #             # 英文处理(去_)
    #             token = text
    #             text_postprocessed = text
    #             result_i = {"key": key[i], "token": token, "text": text_postprocessed}
    #             results.append(result_i)
    #
    #             if ibest_writer is not None:
    #                 ibest_writer["token"][key[i]] = token
    #                 ibest_writer["text"][key[i]] = text_postprocessed
    #
    #             # 中文处理(加空格)
    #             # text_postprocessed, _ = postprocess_utils.sentence_postprocess(token)
    #             # result_i = {"key": key[i], "token": token, "text": text_postprocessed}
    #             # results.append(result_i)
    #             #
    #             # if ibest_writer is not None:
    #             #     ibest_writer["token"][key[i]] = " ".join(token)
    #             #     ibest_writer["text"][key[i]] = text_postprocessed
    #
    #     return results, meta_data


    '''wenet style inference'''
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

        # init beamsearch
        if self.beam_search is None:
            logging.info("enable beam_search")
            self.init_beam_search(**kwargs)
            self.nbest = kwargs.get("nbest", 1)

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
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        pny, pny_lengths = self.ctc_out(encoder_out, encoder_out_lens, key)
        pny = pny.masked_fill(pny == -1, self.eos)
        if len(pny_lengths.size()) > 1:
            pny_lengths = pny_lengths[:, 0]

        # 2. Text Encoder
        text_encoder_out, text_encoder_out_lens = self.text_encode(pny, pny_lengths)

        results = []
        b, _, _ = text_encoder_out.size()
        if isinstance(key[0], (list, tuple)):
            key = key[0]
        if len(key) < b:
            key = key * b
        for i in range(b):
            nbest_idx = 0
            ibest_writer = None
            if kwargs.get("output_dir") is not None:
                if not hasattr(self, "writer"):
                    self.writer = DatadirWriter(kwargs.get("output_dir"))
                ibest_writer = self.writer[f"{nbest_idx+1}best_recog"]

            # 单解码
            beam_size = 10
            length_penalty = 0.0
            text_encoder_mask = (~make_pad_mask(text_encoder_out_lens)[:, None, :]).to(text_encoder_out.device)
            encoder_mask = (~make_pad_mask(encoder_out_lens)[:, None, :]).to(encoder_out.device)
            second_results = mix_beam_search(self, encoder_out, encoder_mask, text_encoder_out, text_encoder_mask)
            token_int = second_results[0].tokens

            # 双重解码
            # beam_size = 10
            # length_penalty = 0.0
            # text_encoder_mask = (~make_pad_mask(text_encoder_out_lens)[:, None, :]).to(text_encoder_out.device)
            # results = attention_beam_search(self, text_encoder_out, text_encoder_mask, beam_size, length_penalty)
            #
            # # 2次解码
            # second_token_int = results[0].tokens
            # second_token = "".join(self.tokenizer.ids2tokens(second_token_int))
            # second_pny_id = self.han2pny(second_token)
            #
            # second_pny = torch.tensor(second_pny_id, dtype=torch.int64).unsqueeze(0).to(device=kwargs["device"])
            # second_pny_lengths = torch.tensor([second_pny.size(1)], dtype=torch.int32).to(device=kwargs["device"])
            #
            # second_text_encoder_out, second_text_encoder_out_lens = self.text_encode(second_pny, second_pny_lengths)
            # second_text_encoder_mask = (~make_pad_mask(second_text_encoder_out_lens)[:, None, :]).to(second_text_encoder_out.device)
            # encoder_mask = (~make_pad_mask(encoder_out_lens)[:, None, :]).to(encoder_out.device)
            # second_results = mix_beam_search(self, encoder_out, encoder_mask, second_text_encoder_out, second_text_encoder_mask)
            #
            # token_int = second_results[0].tokens

            if tokenizer is not None:
                # Change integer-ids to tokens
                token = tokenizer.ids2tokens(token_int)
                text_postprocessed = tokenizer.tokens2text(token)
                if not hasattr(tokenizer, "bpemodel"):
                    text_postprocessed, _ = postprocess_utils.sentence_postprocess(token)

                result_i = {"key": key[i], "text": text_postprocessed}

                if ibest_writer is not None:
                    ibest_writer["token"][key[i]] = " ".join(token)
                    # ibest_writer["text"][key[i]] = text
                    ibest_writer["text"][key[i]] = text_postprocessed
            else:
                result_i = {"key": key[i], "token_int": token_int}
            results.append(result_i)

        return results, meta_data
