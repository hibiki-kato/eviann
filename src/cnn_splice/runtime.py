"""Vendored/adapted dataset, evaluation, and checkpoint-loading helpers.

- ``normalize_checkpoint_state_dict``: verbatim from
  ``dev/IntronModel/src/util/model_runtime.py:1077-1090``.
- ``stratified_split``: verbatim from
  ``dev/IntronModel/src/models/cnn.py:427-445``.
- ``evaluate``: adapted (trimmed to loss/acc@0.5, dropping AMP/ROC/PR-AUC/F1
  machinery not needed by this narrow CPU-oriented driver) from
  ``dev/IntronModel/src/models/cnn.py:448-537``.
- ``DNADataset``: adapted from ``dev/IntronModel/src/models/cnn.py:373-420``.
- ``pick_device``: minimal fresh helper (cuda if available, else cpu);
  not vendored, since IntronModel's ``pick_device`` also resolves mps/amp
  concerns unrelated to this driver.
"""

from __future__ import annotations

import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .arch import one_hot_encode_dna


def normalize_checkpoint_state_dict(
    raw_state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize legacy/compiled checkpoint keys for plain model loading.

    Verbatim from ``dev/IntronModel/src/util/model_runtime.py:1077-1090``.
    """
    compiled_prefix = "_orig_mod."
    normalized: dict[str, torch.Tensor] = {}
    for key, value in raw_state_dict.items():
        if not key.startswith(compiled_prefix):
            normalized[key] = value
    for key, value in raw_state_dict.items():
        if key.startswith(compiled_prefix):
            stripped_key = key[len(compiled_prefix):]
            normalized.setdefault(stripped_key, value)
    return normalized


def pick_device(preference: str = "auto") -> str:
    """Resolve a torch device string. Fresh helper (cuda-or-cpu only)."""
    if preference not in ("auto", ""):
        return preference
    return "cuda" if torch.cuda.is_available() else "cpu"


class DNADataset(Dataset):
    """One-hot encoded DNA sequence dataset.

    Adapted from ``dev/IntronModel/src/models/cnn.py:373-420`` (drops the
    ``preencode``-for-mps special case, which is not relevant on CPU/CUDA).
    """

    def __init__(self, examples: Sequence[Tuple[str, int]], window_len: int) -> None:
        self.examples: list[Tuple[str, int]] = list(examples)
        self.window_len = window_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq, label = self.examples[idx]
        x = one_hot_encode_dna(seq, self.window_len)
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.float32)


def stratified_split(
    examples: Sequence[Tuple[str, int]], val_frac: float = 0.2, seed: int = 1337
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Verbatim from ``dev/IntronModel/src/models/cnn.py:427-445``."""
    rng = random.Random(seed)
    pos = [(s, y) for s, y in examples if y == 1]
    neg = [(s, y) for s, y in examples if y == 0]

    rng.shuffle(pos)
    rng.shuffle(neg)

    n_val_pos = max(1, int(len(pos) * val_frac))
    n_val_neg = max(1, int(len(neg) * val_frac))

    train = pos[n_val_pos:] + neg[n_val_neg:]
    val = pos[:n_val_pos] + neg[:n_val_neg]

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    loss_fn: nn.Module,
) -> Dict[str, float]:
    """Evaluate a model's loss and acc@0.5 on a loader.

    Adapted (AMP/ROC-AUC/PR-AUC/max-F1 stripped) from
    ``dev/IntronModel/src/models/cnn.py:448-537``.
    """
    model.eval()
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    loss_total = 0.0
    loss_examples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss_total += float(loss.detach().float().item()) * int(y.numel())
        loss_examples += int(y.numel())
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(y.float().cpu().numpy())

    logits = np.concatenate(all_logits) if all_logits else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    probs = 1.0 / (1.0 + np.exp(-logits)) if logits.size else np.array([])
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    labels_int = labels.astype(np.int32)

    metrics: Dict[str, float] = {}
    if loss_examples:
        metrics["loss"] = loss_total / loss_examples
    if labels_int.size:
        metrics["acc@0.5"] = float(np.mean((probs >= 0.5) == (labels_int >= 0.5)))
    return metrics
