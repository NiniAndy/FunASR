import logging
from typing import Union, Dict, List, Tuple, Optional

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.cuda.amp import autocast
import re
from funasr.models.scama.utils import sequence_mask
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.models.ctc.ctc import CTC
from funasr.models.transformer.utils.add_sos_eos import add_sos_eos
from funasr.metrics.compute_acc import th_accuracy, compute_accuracy
from funasr.metrics.common import ErrorCalculator
from funasr.train_utils.device_funcs import force_gatherable
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.utils import postprocess_utils
from funasr.utils.datadir_writer import DatadirWriter
from funasr.register import tables
from funasr.train_utils.device_funcs import to_device
import traceback
from torch.nn.utils.rnn import pad_sequence


from typing import List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel, AutoModelForSeq2SeqLM, T5ForConditionalGeneration

dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@tables.register("model_classes", "WhisperParaformer")
class WhisperParaformer(nn.Module):
    """ """

    def __init__(
        self,
        encoder: str = None,
        encoder_conf: dict = None,
        ctc: str = None,
        ctc_conf: Optional[Dict] = None,
        decoder: str = None,
        decoder_conf: Optional[Dict] = None,
        vocab_size: int = -1,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        **kwargs,
    ):

        super().__init__()

        # encoder hub == 'ms'
        from funasr import AutoModel
        whisper = AutoModel(model=encoder, model_revision="master")
        # frontend = model.kwargs.get("frontend")
        encoder_output_size = whisper.model.encoder_output_size
        encoder = whisper.model.model.encoder
        freeze = encoder_conf.get("freeze", True)
        if freeze:
            for name, param in encoder.named_parameters():
                param.requires_grad = False
            encoder.eval()
        self.encoder = encoder
        self.encoder_freeze = freeze

        # uni_phone_ctc
        ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **ctc_conf)
        self.ctc = ctc

        # ldt module


        # paraformer V2 decoder
        decoder_class = tables.decoder_classes.get(decoder)
        decoder = decoder_class(vocab_size=vocab_size, encoder_output_size=encoder_output_size, **decoder_conf,)
        self.decoder = decoder


        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1

        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is not None:
            self.token2id = tokenizer.token2id
            self.blank_id = self.token2id.get("<blank>", blank_id)
            self.sos = self.token2id.get("<s>", self.sos)
            self.eos = self.token2id.get("</s>", self.eos)

            if hasattr(tokenizer, 'add_special_token_list'):
                add_special_token_list = tokenizer.add_special_token_list
            else:
                add_special_token_list = False
            if add_special_token_list:
                self.start_id_of_special_tokens = len(self.token2id) - len(add_special_token_list)
        else:
            self.token2id = None

        self.vocab_size = vocab_size

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None
        self.error_calculator = None
        self.total_token_num, self.error_num = 1, 0
        self.ignore_id = ignore_id

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
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

        # whisper encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        loss_ctc, cer_ctc = None, None
        stats = dict()

        # decoder: CTC branch
        loss_ctc, cer_ctc = self._calc_ctc_loss(encoder_out, encoder_out_lens, text, text_lengths)
        # Collect CTC branch stats
        stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
        stats["cer_ctc"] = cer_ctc

        # decoder: Attention decoder branch
        loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(encoder_out, encoder_out_lens, text, text_lengths)

        loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
        # loss = loss_ctc

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att

        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = (text_lengths + self.predictor_bias).sum()
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight




    def encode(self, speech, speech_lengths, **kwargs,):
        speech = speech.permute(0, 2, 1)
        res = self.encoder(speech)
        if isinstance(res, (list, tuple)):
            encoder_out, encoder_out_lens = res[0], res[1]
        else:
            encoder_out, encoder_out_lens = res, speech_lengths
        return encoder_out, encoder_out_lens


    def _calc_ctc_loss(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens,):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc



    def _calc_att_loss(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens,):

        batch_size = encoder_out.size(0)
        with torch.no_grad():
            compressed_ctc_batch  = []
            ctc_probs = self.ctc.log_softmax(encoder_out).detach()

            for b in range(batch_size):
                ctc_prob = ctc_probs[b][: encoder_out_lens[b]].cpu() # [T, N]
                text_b = ys_pad[b][: ys_pad_lens[b]].cpu() # [1, U]
                text_audio_alignment = self.ctc.force_align(ctc_prob, text_b)
                text_audio_alignment = torch.tensor(text_audio_alignment)
                audio_text = self.ctc.remove_duplicates_and_blank(text_audio_alignment, self.blank_id)
                if len(audio_text) != ys_pad_lens[b]:
                    print (f"ctc alignment error: {audio_text}, {text_b}")
                # 把相同的不为0的帧的概率平均
                ctc_comp = self.average_repeats(ctc_prob, text_audio_alignment)
                if ctc_comp.size(0) != ys_pad_lens[b]:
                    print (f"ctc_comp error: {ctc_comp.size(0)}, {text_b}")
                compressed_ctc_batch.append(ctc_comp)

            padded_ctc_batch = pad_sequence(compressed_ctc_batch, batch_first=True).to(encoder_out.device)

        decoder_outs = self.decoder(encoder_out, encoder_out_lens, padded_ctc_batch, ys_pad_lens)
        decoder_out, _ = decoder_outs[0], decoder_outs[1]


        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_pad)
        acc_att = th_accuracy(decoder_out.view(-1, self.vocab_size),  ys_pad, ignore_label=self.ignore_id,)

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

