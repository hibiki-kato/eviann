"""Vendored binary focal loss (the loss configured for cnn_v4.3 training).

``dev/IntronModel/run/.train_v43.sh`` sets ``LOSS="focal"`` and
``FOCAL_GAMMA="2.0"``. This module vendors the real ``BinaryFocalLoss`` and
``_resolve_alpha_pos``/``build_binary_classification_loss`` logic from
``dev/IntronModel/src/util/losses.py``:

- ``BinaryFocalLoss``: verbatim from losses.py:53-126
- ``_resolve_alpha_pos``: verbatim from losses.py:354-367
- ``build_focal_loss``: adapted from the ``loss_name == "focal"`` branch of
  ``build_binary_classification_loss``, losses.py:452-472

Other loss variants (bce, weighted_bce, asymmetric_focal, f1, and the mixed
losses) are intentionally left out: they are not used by the tuned cnn_v4.3
config for this pipeline.
"""

from __future__ import annotations

from typing import Optional, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import sigmoid_focal_loss as torchvision_sigmoid_focal_loss
except ImportError:  # pragma: no cover
    torchvision_sigmoid_focal_loss = None


class LossMeta(TypedDict):
    """Metadata for configured loss behavior."""

    pos_weight: float
    focal_gamma: float
    focal_alpha_pos: float


class BinaryFocalLoss(nn.Module):
    """Binary focal loss for imbalanced classification.

    Verbatim from ``dev/IntronModel/src/util/losses.py:53-126``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha_pos: float = 0.75,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0.0:
            raise ValueError("gamma must be >= 0.0")
        if not (0.0 < alpha_pos < 1.0):
            raise ValueError("alpha_pos must satisfy 0 < alpha_pos < 1")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: mean, sum, none")
        self.gamma = gamma
        self.alpha_pos = alpha_pos
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        if torchvision_sigmoid_focal_loss is not None:
            return torchvision_sigmoid_focal_loss(
                inputs=logits,
                targets=targets,
                alpha=self.alpha_pos,
                gamma=self.gamma,
                reduction=self.reduction,
            )

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha_pos * targets + (1.0 - self.alpha_pos) * (1.0 - targets)
        focal_factor = torch.pow(1.0 - pt, self.gamma)
        loss = alpha_t * focal_factor * bce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def _resolve_alpha_pos(
    alpha_pos: Optional[float],
    train_pos: int,
    train_neg: int,
) -> float:
    """Resolve positive-class alpha from argument or class imbalance.

    Verbatim from ``dev/IntronModel/src/util/losses.py:354-367``.
    """
    if alpha_pos is not None:
        resolved = float(alpha_pos)
    else:
        total = max(1, train_pos + train_neg)
        resolved = float(train_neg / total)
    if not (0.0 < resolved < 1.0):
        raise ValueError("Resolved alpha_pos must satisfy 0 < alpha_pos < 1")
    return resolved


def build_focal_loss(
    *,
    train_pos: int,
    train_neg: int,
    focal_gamma: float = 2.0,
    focal_alpha_pos: Optional[float] = None,
) -> tuple[nn.Module, LossMeta]:
    """Build the configured focal loss, adapted from the ``"focal"`` branch of
    ``build_binary_classification_loss`` (``dev/IntronModel/src/util/losses.py:452-472``).
    """
    alpha_pos = _resolve_alpha_pos(
        alpha_pos=focal_alpha_pos, train_pos=train_pos, train_neg=train_neg,
    )
    criterion = BinaryFocalLoss(gamma=float(focal_gamma), alpha_pos=alpha_pos, reduction="mean")
    return criterion, LossMeta(
        pos_weight=1.0,
        focal_gamma=float(focal_gamma),
        focal_alpha_pos=alpha_pos,
    )
