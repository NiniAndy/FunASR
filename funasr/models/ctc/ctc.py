import logging

import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import numpy as np

class CTC(nn.Module):
    """CTC module.

    Args:
        odim: dimension of outputs
        encoder_output_size: number of encoder projection units
        dropout_rate: dropout rate (0.0 ~ 1.0)
        ctc_type: builtin or warpctc
        reduce: reduce the CTC loss into a scalar
    """

    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        ctc_type: str = "builtin",
        reduce: bool = True,
        ignore_nan_grad: bool = True,
        blank_id = 0,
    ):
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = nn.Linear(eprojs, odim)
        self.ctc_type = ctc_type
        self.ignore_nan_grad = ignore_nan_grad

        if self.ctc_type == "builtin":
            reduction_type = "sum" if reduce else "none"
            self.ctc_loss = torch.nn.CTCLoss(blank=blank_id, reduction=reduction_type, zero_infinity=True)

        elif self.ctc_type == "warpctc":
            import warpctc_pytorch as warp_ctc
            if ignore_nan_grad:
                logging.warning("ignore_nan_grad option is not supported for warp_ctc")
            self.ctc_loss = warp_ctc.CTCLoss(size_average=True, reduce=reduce)
        else:
            raise ValueError(f'ctc_type must be "builtin" or "warpctc": {self.ctc_type}')

        self.reduce = reduce

    def forward(self, hs_pad: torch.Tensor, hlens: torch.Tensor,
                ys_pad: torch.Tensor,
                ys_lens: torch.Tensor):
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        # ys_hat: (B, L, D) -> (L, B, D)
        ys_hat = ys_hat.transpose(0, 1)
        ys_hat = ys_hat.log_softmax(2)
        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
        # Batch-size average
        loss = loss / ys_hat.size(1)
        ys_hat = ys_hat.transpose(0, 1)
        return loss

    def softmax(self, hs_pad):
        """softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.softmax(self.ctc_lo(hs_pad), dim=2)

    def log_softmax(self, hs_pad):
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad):
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)

    def ctc_logprobs(self, hs_pad, blank_id=0, blank_penalty: float = 0.0,):
        """log softmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        if blank_penalty > 0.0:
            logits = self.ctc_lo(hs_pad)
            logits[:, :, blank_id] -= blank_penalty
            ctc_probs = logits.log_softmax(dim=2)
        else:
            ctc_probs = self.log_softmax(hs_pad)
        return ctc_probs

    def insert_blank(self, label, blank_id=0):
        """Insert blank token between every two label token."""
        label = np.expand_dims(label, 1)
        blanks = np.zeros((label.shape[0], 1), dtype=np.int64) + blank_id
        label = np.concatenate([blanks, label], axis=1)
        label = label.reshape(-1)
        label = np.append(label, label[0])
        return label

    def force_align(self, ctc_probs: torch.Tensor, y: torch.Tensor, blank_id=0) -> list:
        """ctc forced alignment.

        Args:
            torch.Tensor ctc_probs: hidden state sequence, 2d tensor (T, D)
            torch.Tensor y: id sequence tensor 1d tensor (L)
            int blank_id: blank symbol index
        Returns:
            torch.Tensor: alignment result
        """
        ctc_probs = ctc_probs[None].cpu()
        y = y[None].cpu()
        alignments, _ = torchaudio.functional.forced_align(ctc_probs, y, blank=blank_id)
        return alignments[0]

    def remove_duplicates_and_blank(self, alignment, blank_id=0):
        """
        去除对齐路径中的空白标签和重复标签。

        alignment: 对齐路径，可能包含空白标签和重复标签。
        blank_id: 空白标签的 ID。

        返回：
        filtered_alignment: 去除空白和重复标签后的对齐路径。
        """
        filtered_alignment = []
        prev_token = None
        for token in alignment:
            if token != blank_id and token != prev_token:
                filtered_alignment.append(token)
            prev_token = token
        return filtered_alignment

    def remove_duplicates(self, alignment):
        """
        去除对齐路径中的空白标签和重复标签。

        alignment: 对齐路径，可能包含空白标签和重复标签。
        blank_id: 空白标签的 ID。

        返回：
        filtered_alignment: 去除空白和重复标签后的对齐路径。
        """
        filtered_alignment = []
        prev_token = None
        for token in alignment:
            if  token != prev_token:
                filtered_alignment.append(token)
            prev_token = token
        return filtered_alignment
