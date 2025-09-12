# ntp_smtp_loss.py
"""Loss helpers for self‑supervised TokenGT training (NTP / SMTP).

* NTP (Next‑Token Prediction): standard LM cross‑entropy on each step.
* SMTP (Scheduled Masked‑Token Prediction): progressively increase mask
  ratio; only masked positions contribute to loss.
"""
from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

LossType = Literal["ntp", "smtp"]


def ntp_loss(logits: Tensor, target: Tensor, ignore_index: int = -100) -> Tensor:
    """Cross‑entropy for next‑token prediction.

    * logits: `[T, B, V]`  (output from model)
    * target: `[T, B]`     (ground‑truth next token)
    """
    logp = logits.view(-1, logits.size(-1))
    tgt = target.view(-1)
    return F.cross_entropy(logp, tgt, ignore_index=ignore_index)


def smtp_loss(
    logits: Tensor,
    target: Tensor,
    mask_label: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """Masked‑token prediction loss.

    * logits: `[T, B, V]`
    * target: `[T, B]`  – same shape as mask_label
    * mask_label: bool/int tensor (1 => token was masked)
    Only positions with mask_label==1 are included.
    """
    assert target.shape == mask_label.shape == logits.shape[:2]

    logp = logits.view(-1, logits.size(-1))
    tgt = target.view(-1)
    weights = mask_label.view(-1).to(logits.dtype)

    loss = F.cross_entropy(logp, tgt, reduction="none", ignore_index=ignore_index)
    loss = (loss * weights).sum() / (weights.sum() + 1e-6)
    return loss


class ScheduledMasker:
    """Linear schedule mask generator (ratio ↑ per epoch).

    Not tied to loss, but provided here for convenience.
    """

    def __init__(self, start_ratio: float = 0.1, end_ratio: float = 0.8, epochs: int = 30):
        self.start = start_ratio
        self.end = end_ratio
        self.epochs = max(1, epochs)

    def ratio(self, current_epoch: int) -> float:
        t = min(current_epoch, self.epochs)
        return self.start + (self.end - self.start) * t / self.epochs

    def __call__(self, input_ids: Tensor, ratio: float) -> Tuple[Tensor, Tensor]:
        """Return `(masked_input, mask_label)`.

        * input_ids: `[T, B]`
        """
        device = input_ids.device
        mask = torch.rand_like(input_ids, dtype=torch.float, device=device) < ratio
        masked_input = input_ids.clone()
        masked_input[mask] = 0  # assuming 0 == [MASK]
        return masked_input, mask.to(input_ids.dtype)
