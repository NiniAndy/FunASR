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
from funasr.models.transformer.search_method import attention_beam_search
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from jiwer import cer, wer


@tables.register("model_classes", "SLM")
class SLM(nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        decoder: str = None,
        decoder_conf: dict = None,
        vocab_size: int = -1,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        share_embedding: bool = False,
        **kwargs,
    ):

        super().__init__()

        decoder_class = tables.decoder_classes.get(decoder)
        decoder = decoder_class(vocab_size=vocab_size,  **decoder_conf)


        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id

        self.tokenizer = kwargs["tokenizer"]
        self.placeholder_id = self.tokenizer.token2id["<phd>"]

        # self.error_calculator = None
        self.decoder = decoder

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
        pny: torch.Tensor,
        pny_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Encoder + Decoder + Calc loss
        Args:
                pny: (Batch, Length)
                pny_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(pny_lengths.size()) > 1:
            pny_lengths = pny_lengths[:, 0]

        batch_size = pny.shape[0]

        stats = dict()

        # decoder: Attention decoder branch
        loss_att, acc_att= self._calc_att_loss(pny, pny_lengths, text, text_lengths)
        loss = loss_att


        stats["acc"] = acc_att
        stats["loss"] = torch.clone(loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((text_lengths + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight


    def _calc_att_loss(
        self,
        pny: torch.Tensor,
        pny_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):

        pny = pny.masked_fill(pny == -1, self.placeholder_id)
        ys_in_pad, ys_out_pad = add_sos_eos(text, self.sos, self.eos, self.ignore_id)
        text_lengths = text_lengths + 1
        decoder_input = torch.cat([pny, ys_in_pad], dim=-1)

        decoder_out, _ = self.decoder(decoder_input, text_lengths, pny_lengths)

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        return loss_att, acc_att

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

        # c. Passed the encoder result and the beam search
        nbest_hyps = self.beam_search(
            x=encoder_out[0],
            maxlenratio=kwargs.get("maxlenratio", 0.0),
            minlenratio=kwargs.get("minlenratio", 0.0),
        )

        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        b, n, d = encoder_out.size()
        for i in range(b):

            for nbest_idx, hyp in enumerate(nbest_hyps):
                ibest_writer = None
                if kwargs.get("output_dir") is not None:
                    if not hasattr(self, "writer"):
                        self.writer = DatadirWriter(kwargs.get("output_dir"))
                    ibest_writer = self.writer[f"{nbest_idx + 1}best_recog"]

                # remove sos/eos and get results
                last_pos = -1
                if isinstance(hyp.yseq, list):
                    token_int = hyp.yseq[1:last_pos]
                else:
                    token_int = hyp.yseq[1:last_pos].tolist()

                # remove blank symbol id, which is assumed to be 0
                token_int = list(
                    filter(
                        lambda x: x != self.eos and x != self.sos and x != self.blank_id, token_int
                    )
                )

                # Change integer-ids to tokens
                token = tokenizer.ids2tokens(token_int)
                text = tokenizer.tokens2text(token)
                # 英文处理(去_)
                token = text
                text_postprocessed = text
                result_i = {"key": key[i], "token": token, "text": text_postprocessed}
                results.append(result_i)

                if ibest_writer is not None:
                    ibest_writer["token"][key[i]] = token
                    ibest_writer["text"][key[i]] = text_postprocessed

                # 中文处理(加空格)
                # text_postprocessed, _ = postprocess_utils.sentence_postprocess(token)
                # result_i = {"key": key[i], "token": token, "text": text_postprocessed}
                # results.append(result_i)
                #
                # if ibest_writer is not None:
                #     ibest_writer["token"][key[i]] = " ".join(token)
                #     ibest_writer["text"][key[i]] = text_postprocessed

        return results, meta_data


    def test(
        self,
        pny: torch.Tensor,
        pny_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,

    ):
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(pny_lengths.size()) > 1:
            pny_lengths = pny_lengths[:, 0]

        batch_size = pny.shape[0]
        if batch_size > 1:
            raise NotImplementedError("batch decoding is not implemented")

        # 1. Encoder
        # 把所有的-1变成2
        pny = pny.masked_fill(pny == -1, 2)
        encoder_out, encoder_out_lens = self.encode(pny, pny_lengths)

        beam_size = 10
        length_penalty = 0.0
        encoder_mask = (~make_pad_mask(encoder_out_lens)[:, None, :]).to(encoder_out.device)
        results = attention_beam_search(self, encoder_out, encoder_mask, beam_size, length_penalty)
        token_int = results[0].tokens

        if self.tokenizer is not None:
            # Change integer-ids to tokens
            token = self.tokenizer.ids2tokens(token_int)
            text_postprocessed = self.tokenizer.tokens2text(token)
            if not hasattr(self.tokenizer, "bpemodel"):
                text_postprocessed, _ = postprocess_utils.sentence_postprocess(token)

        ground_truth = self.tokenizer.ids2tokens(text[0].tolist())
        ground_truth = self.tokenizer.tokens2text(ground_truth)
        error = cer(ground_truth, text_postprocessed)
        total_len = len(ground_truth)
        error_num = error * total_len

        return {"cer": error, "total_len": total_len, "error_num": error_num, "text": text_postprocessed, "ground_truth": ground_truth}

