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
from funasr.models.accent_recognition.lasas import LASAS
from funasr.models.accent_recognition.asd import ASD

import os
import yaml
from funasr.auto.auto_model import AutoModel


@tables.register("model_classes", "TransformerAnchorContrastAr")
class TransformerAnchorContrastAr(nn.Module):
    """识别口音"""

    def __init__(
        self,
        specaug: str = None,
        specaug_conf: dict = None,
        normalize: str = None,
        normalize_conf: dict = None,
        encoder: str = None,
        encoder_conf: dict = None,
        decoder: str = None,
        decoder_conf: dict = None,
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

        if normalize is not None:
            normalize_class = tables.normalize_classes.get(normalize)
            normalize = normalize_class(**normalize_conf)

        encoder_class = tables.encoder_classes.get(encoder)
        encoder = encoder_class(input_size=input_size, **encoder_conf)
        encoder_output_size = encoder.output_size()

        text_language_vocab_path = kwargs.get("text_language_vocab_path", None)
        assert text_language_vocab_path is not None, "text_language_vocab_path is required"
        text_language_vocab = {}
        with open(kwargs['text_language_vocab_path'], "r") as f:
            for line in f:
                line = line.strip()
                text_language_vocab[line] = len(text_language_vocab)

        dialect_size = len(text_language_vocab)

        asd_conf = kwargs.get("asd_conf", {})
        # if len(encoder.interctc_layer_idx) != 0:
        #     layer_combine_num = len(encoder.interctc_layer_idx)
        #     lasas_input_size = encoder_output_size * layer_combine_num
        # else:
        #     lasas_input_size = encoder_output_size
        asd_input_size = encoder_output_size
        acc_num = asd_conf.get("accent_size", dialect_size)
        num_space = len(encoder.interctc_layer_idx)

        asd = ASD(asd_input_size, acc_num, num_spaces=num_space, **asd_conf)

        asd_output_size = asd.output_size()

        # merge_encoder_conf = kwargs.get("merge_encoder_conf", {})
        # if merge_encoder_conf is not None:
        #     merge_encoder_conf["input_size"] = encoder_output_size + asd_output_size
        #     merge_encoder_conf["num_blocks"] = 3
        #     merge_encoder_conf["interctc_layer_idx"] = []
        #     merge_encoder = encoder_class(input_size=encoder_output_size + asd_output_size, **merge_encoder_conf)
        #     self.merge_encoder = merge_encoder
        # merge_encoder_conf["num_blocks"] = 3
        # merge_encoder_conf["interctc_layer_idx"] = []
        # merge_encoder = encoder_class(input_size=encoder_output_size + asd_output_size, **merge_encoder_conf)
        # self.merge_encoder = merge_encoder

        decoder_input_size = encoder_output_size

        if decoder is not None:
            decoder_class = tables.decoder_classes.get(decoder)
            decoder = decoder_class(
                vocab_size=vocab_size,
                encoder_output_size=decoder_input_size,
                **decoder_conf
            )

        anchor_conf = kwargs["anchor_conf"]
        anchor_base_path = anchor_conf["config_path"]
        anchor_path = os.path.join(anchor_base_path, "config.yaml")
        with open(anchor_path, "r") as f:
            anchor_kwargs = yaml.safe_load(f)
        # 屏蔽无关参数
        anchor_kwargs["model_conf"]["ctc_weight"] = 0.0
        anchor_kwargs["tokenizer"] = None
        anchor_kwargs["tokenizer_conf"] = None
        anchor_kwargs["specaug"] = None
        anchor_kwargs["specaug_conf"] = None
        anchor_kwargs["decoder"]= None
        anchor_kwargs["decoder_conf"] = None
        anchor_kwargs["train_conf"] = None
        anchor_kwargs["optim"] = None
        anchor_kwargs["optim_conf"] = None
        anchor_kwargs["scheduler"] = None
        anchor_kwargs["scheduler_conf"] = None
        anchor_kwargs["dataset"] = None
        anchor_kwargs["dataset_conf"] = None
        anchor_kwargs["ctc_conf"] = None
        anchor_kwargs["init_param"] = os.path.join(anchor_base_path, "model.pt")
        if len(encoder.interctc_layer_idx) != 0:
            anchor_kwargs["encoder_conf"]["interctc_layer_idx"] = encoder.interctc_layer_idx

        anchor_model = AutoModel(**anchor_kwargs).model.eval()
        delattr(anchor_model, 'criterion_att')
        self.anchor_model = anchor_model
        for param in self.anchor_model.parameters():
            param.requires_grad = False

        if ctc_weight > 0.0:

            if ctc_conf is None:
                ctc_conf = {}

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
        self.ctc_weight = ctc_weight
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.asd = asd

        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )
        self.interctc_weight = interctc_weight

        # self.error_calculator = None
        if ctc_weight == 1.0:
            self.decoder = None
        else:
            self.decoder = decoder

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        #
        # if report_cer or report_wer:
        #     self.error_calculator = ErrorCalculator(
        #         token_list, sym_space, sym_blank, report_cer, report_wer
        #     )
        #
        self.error_calculator = None
        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

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
        anchor = kwargs["anchor"]

        anchor_out, _  = self.anchor_model.encode(anchor, speech_lengths)
        intermediate_anchor = None
        if isinstance(anchor_out, tuple):
            intermediate_anchor = anchor_out[1]
            anchor_out = anchor_out[0]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # 拼接Xa
        if intermediate_outs is not None:
            intermediate_out_list = [out.unsqueeze(1) for _, out in intermediate_outs]
            X_text = torch.cat(intermediate_out_list, dim=1)
            intermediate_anchor_list = [out.unsqueeze(1) for _, out in intermediate_anchor]
            X_pron = torch.cat(intermediate_anchor_list, dim=1)

        else:
            X_text = encoder_out.unsqueeze(1)
            X_pron = anchor_out.unsqueeze(1)

        Xar, _ = self.asd(X_text, X_pron, encoder_out_lens)

        loss_dal, acc_dal, dal_out = None, None, None
        # 有效计数
        effective_count = text_language.ne(-1).sum().item()



        loss_dal, acc_dal, dal_out = self.asd.calculate_loss(Xar, text_language)


        # decoder_input = torch.cat([encoder_out, Xdnn], dim=-1)
        decoder_input = encoder_out
        decoder_input_lens = encoder_out_lens


        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        stats = dict()

        # decoder: CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out
                loss_ic, cer_ic = self._calc_ctc_loss(
                    intermediate_out, encoder_out_lens, text, text_lengths
                )
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                stats["loss_interctc_layer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            loss_ctc = (1 - self.interctc_weight) * loss_ctc + self.interctc_weight * loss_interctc

        # decoder: Attention decoder branch
        langauge_emb = Xar.clone()
        loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
            decoder_input, decoder_input_lens, text, text_lengths, langauge_emb, text_language
        )

        # 3. CTC-Att loss definition
        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        # 如果loss_dal不是Nan则加入到loss中
        if loss_dal is not None:
            loss = loss + loss_dal


        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att
        stats["loss_dal"] = loss_dal.detach() if loss_dal is not None else None
        stats["acc_dal"] = acc_dal
        stats["eff_num"] = effective_count

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

            # 如果encoder的embed是Hubert/Wav2Vec2的embed则需要先进行cnn再做增强
            if isinstance(self.encoder.embed, HubertFeatureEncoder):
                speech = speech.transpose(1, 2)
                speech, speech_lengths = self.encoder.feature_extractor_forward(speech, speech_lengths)
                speech, pos_emb = speech
                speech, speech_lengths = self.specaug(speech, speech_lengths)
                speech = (speech, pos_emb)
                if self.encoder.interctc_use_conditioning:
                    encoder_out, encoder_out_lens, _ = self.encoder.encoder_forward(speech, speech_lengths, ctc=self.ctc)
                else:
                    encoder_out, encoder_out_lens, _ = self.encoder.encoder_forward(speech, speech_lengths)
            else:
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


    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        text_language: torch.Tensor,
        text_language_label: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens, text_language, text_language_label=text_language_label)

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
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

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

                text_postprocessed, _ = postprocess_utils.sentence_postprocess(token)
                result_i = {"key": key[i], "token": token, "text": text_postprocessed}
                results.append(result_i)

                if ibest_writer is not None:
                    ibest_writer["token"][key[i]] = " ".join(token)
                    ibest_writer["text"][key[i]] = text_postprocessed

        return results, meta_data