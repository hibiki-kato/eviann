"""Vendored CNN v4.3 architecture (donor/acceptor splice-site classifier).

This is a narrowed, self-contained copy of the real cnn_v4 architecture used
by ``dev/IntronModel``. Classes below are copied/adapted verbatim from:

- ``dev/IntronModel/src/models/cnn_common.py``
  (``one_hot_encode_dna``, ``normalize_cnn_head_type``,
  ``_readout_sequence_features``, lines ~49-209)
- ``dev/IntronModel/src/models/cnn_pair_v3.py``
  (``OrganicBranchLayout``, ``ResidualDilatedBlock``,
  ``ResidualDilatedBranchEncoder``, lines ~182-478)
- ``dev/IntronModel/src/models/cnn_v4.py``
  (``DeformableConv1d``, ``GroupedDeformableStem``, ``OrganicSiteCNN``,
  lines ~71-229)

Left out on purpose (per the narrower use case): the argparse-namespace
based hyperparameter-resolution indirection (``_resolve_task_arch_params``,
``_resolve_task_deformable_params``), multi-species/HPO plumbing,
torch.compile, checkpoint pruning/versioning, and AMP dtype auto-detection.
Hyperparameters are supplied directly as an ``ArchConfig`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CNN_HEAD_TYPE_CHOICES: tuple[str, ...] = ("gap", "center")


# ---------------------------------------------------------------------------
# Vendored from cnn_common.py
# ---------------------------------------------------------------------------


def one_hot_encode_dna(seq: str, window_len: int = 200) -> np.ndarray:
    """One-hot encode a DNA sequence. Verbatim from cnn_common.py:49-71."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoded = np.zeros((4, window_len), dtype=np.float32)
    for i, base in enumerate(seq[:window_len].upper()):
        if base in mapping:
            encoded[mapping[base], i] = 1.0
    return encoded


def normalize_cnn_head_type(raw: object, *, arg_name: str) -> str:
    """Verbatim from cnn_common.py:147-171."""
    head_type = str(raw).strip().lower()
    if head_type in CNN_HEAD_TYPE_CHOICES:
        return head_type
    choices_text = ", ".join(CNN_HEAD_TYPE_CHOICES)
    raise ValueError(f"{arg_name} must be one of: {choices_text}.")


def _readout_sequence_features(x: torch.Tensor, head_type: str) -> torch.Tensor:
    """Verbatim from cnn_common.py:174-209."""
    if x.ndim != 3:
        raise ValueError("x must have shape (batch, channels, length).")
    if x.shape[2] <= 0:
        raise ValueError("x length dimension must be positive.")
    if head_type == "gap":
        return x.mean(dim=-1)
    center_right = x.shape[2] // 2
    if x.shape[2] % 2 == 1:
        return x[:, :, center_right]
    center_left = center_right - 1
    return 0.5 * (x[:, :, center_left] + x[:, :, center_right])


# ---------------------------------------------------------------------------
# Vendored from cnn_pair_v3.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrganicBranchLayout:
    """Per-branch residual-dilated block layout. Verbatim from cnn_pair_v3.py:182-201."""

    channels: list[int]
    kernel_sizes: list[int]
    dilations: list[int]
    residual_channels: list[int]


class ResidualDilatedBlock(nn.Module):
    """Residual 1D CNN block with dilated convolutions.

    Verbatim from cnn_pair_v3.py:341-426.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        residual_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be positive and odd.")
        if dilation <= 0:
            raise ValueError("dilation must be positive.")
        if residual_channels <= 0:
            raise ValueError("residual_channels must be positive.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        padding = ((kernel_size - 1) // 2) * dilation
        self.conv1 = nn.Conv1d(
            in_channels, residual_channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation,
        )
        self.norm1 = nn.BatchNorm1d(residual_channels)
        self.conv2 = nn.Conv1d(
            residual_channels, out_channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation,
        )
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.projection: nn.Module
        if in_channels == out_channels:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projection(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out + residual)
        return out


class ResidualDilatedBranchEncoder(nn.Module):
    """Residual-dilated encoder for one donor or acceptor sequence.

    Verbatim from cnn_pair_v3.py:429-478.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        layout: OrganicBranchLayout,
        head_type: str,
        dropout: float,
    ) -> None:
        super().__init__()
        if not layout.channels:
            raise ValueError("layout.channels must contain at least one block.")
        self.head_type = normalize_cnn_head_type(head_type, arg_name="head_type")
        self.blocks = nn.ModuleList()

        current_in_channels = in_channels
        for channel, kernel_size, dilation, residual_channels in zip(
            layout.channels,
            layout.kernel_sizes,
            layout.dilations,
            layout.residual_channels,
            strict=True,
        ):
            self.blocks.append(
                ResidualDilatedBlock(
                    in_channels=current_in_channels,
                    out_channels=channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    residual_channels=residual_channels,
                    dropout=dropout,
                )
            )
            current_in_channels = channel
        self.output_dim = current_in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return _readout_sequence_features(x, self.head_type)


# ---------------------------------------------------------------------------
# Vendored from cnn_v4.py
# ---------------------------------------------------------------------------


class DeformableConv1d(nn.Module):
    """A differentiable grouped deformable 1D convolution.

    Verbatim from cnn_v4.py:71-170 (torch-native grid_sample implementation).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if groups <= 0 or in_channels % groups or out_channels % groups:
            raise ValueError("groups must divide both in_channels and out_channels.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.padding = kernel_size // 2
        self.offset = nn.Conv1d(in_channels, groups * kernel_size, kernel_size,
                                padding=self.padding, bias=True)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = (self.in_channels // self.groups) * self.kernel_size
            bound = 1.0 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Inputs must have shape (batch, channels, length).")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}."
            )
        batch_size, _, length = x.shape
        offsets = self.offset(x).view(
            batch_size, self.groups, self.kernel_size, length
        )
        base = torch.arange(length, device=x.device, dtype=x.dtype)
        kernel = torch.arange(self.kernel_size, device=x.device, dtype=x.dtype)
        positions = (
            base.view(1, 1, 1, length)
            + kernel.view(1, 1, self.kernel_size, 1)
            - self.padding
            + offsets
        )
        if length == 1:
            x_coords = torch.zeros_like(positions)
        else:
            x_coords = positions.mul(2.0 / (length - 1)).sub(1.0)
        grid = torch.stack((x_coords, torch.zeros_like(x_coords)), dim=-1)
        grid = grid.reshape(batch_size * self.groups, 1, self.kernel_size * length, 2)
        grouped_x = x.reshape(batch_size * self.groups, self.in_channels // self.groups, 1, length)
        sampled = F.grid_sample(
            grouped_x, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
        )
        sampled = sampled.reshape(
            batch_size, self.groups, self.in_channels // self.groups, self.kernel_size, length,
        )
        grouped_weight = self.weight.reshape(
            self.groups, self.out_channels // self.groups, self.in_channels // self.groups, self.kernel_size,
        )
        output = torch.einsum("bgckl,gock->bgol", sampled, grouped_weight)
        output = output.reshape(batch_size, self.out_channels, length)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)
        return output


class GroupedDeformableStem(nn.Module):
    """Grouped pointwise convolution followed by grouped deformable sampling.

    Verbatim from cnn_v4.py:173-193.
    """

    def __init__(self, *, groups: int, kernel_size: int) -> None:
        super().__init__()
        if groups <= 0 or 4 % groups:
            raise ValueError("deformable_groups must divide the 4 one-hot channels.")
        self.groups = groups
        self.grouped_conv = nn.Conv1d(4, 4, kernel_size=1, groups=groups, bias=False)
        self.deformable_conv = DeformableConv1d(4, 4, kernel_size, groups=groups, bias=False)
        self.norm = nn.BatchNorm1d(4)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.activation(self.norm(self.deformable_conv(self.grouped_conv(x))))


@dataclass(frozen=True)
class ArchConfig:
    """Resolved cnn_v4.3 hyperparameters for one donor or acceptor task.

    Defaults match ``dev/IntronModel/data/tuning/cnn_v4_shared/{donor,acceptor}/
    versions/cnn_v4.3.json`` (identical sampled_params for both tasks in that
    tuned config), falling back to ``dev/IntronModel/run/.train_v43.sh`` CONFIG
    values for anything not present there.
    """

    conv_channels: Sequence[int] = (128, 256, 512)
    kernel_sizes: Sequence[int] = (9, 9, 9)
    block_dilations: Sequence[int] = (1, 6, 12)
    residual_channels: Sequence[int] = (48, 96, 192)
    head_type: str = "gap"
    fc_hidden: int = 128
    dropout: float = 0.15561144760882448
    deformable_groups: int = 4
    deformable_kernel_size: int = 5

    def layout(self) -> OrganicBranchLayout:
        depth = len(self.conv_channels)
        if not (len(self.kernel_sizes) == len(self.block_dilations) == len(self.residual_channels) == depth):
            raise ValueError("conv_channels/kernel_sizes/block_dilations/residual_channels must have equal length.")
        return OrganicBranchLayout(
            channels=list(self.conv_channels),
            kernel_sizes=list(self.kernel_sizes),
            dilations=list(self.block_dilations),
            residual_channels=list(self.residual_channels),
        )


class OrganicSiteCNN(nn.Module):
    """cnn_v3 residual-dilated encoder preceded by a grouped deformable stem.

    This is the actual cnn_v4.3 model: adapted from cnn_v4.py:196-228
    (``OrganicSiteCNN``), combined with cnn_v3.py:213-254's fc head, using
    ``ArchConfig`` in place of the argparse-namespace resolution helpers.
    """

    def __init__(self, *, config: ArchConfig) -> None:
        super().__init__()
        if config.dropout < 0.0 or config.dropout >= 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")
        self.stem = GroupedDeformableStem(
            groups=config.deformable_groups,
            kernel_size=config.deformable_kernel_size,
        )
        self.encoder = ResidualDilatedBranchEncoder(
            in_channels=4,
            layout=config.layout(),
            head_type=config.head_type,
            dropout=config.dropout,
        )
        self.fc = nn.Sequential(
            nn.Linear(self.encoder.output_dim, config.fc_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Inputs must have shape (batch, channels, length).")
        features = self.encoder(self.stem(x.float()))
        return self.fc(features)[:, 0]
