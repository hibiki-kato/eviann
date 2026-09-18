"""Batched window scoring shared by the transcript scorer and the site coprocess."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import ACCEPTOR_CHECKPOINT, CNN_DOWNSTREAM, CNN_UPSTREAM, DONOR_CHECKPOINT
from .arch import one_hot_encode_dna
from .model import load_task_model
from .runtime import pick_device
from .windowing import reshape_site_sequence_4p


class SiteModels:
    """Donor + acceptor checkpoints from one model directory, loaded once."""

    def __init__(self, model_dir: Path, device: str = "auto", batch_size: int = 1024) -> None:
        self.device = pick_device(device)
        self.batch_size = batch_size
        self.donor, donor_ckpt = load_task_model(Path(model_dir) / DONOR_CHECKPOINT, self.device)
        self.acceptor, acceptor_ckpt = load_task_model(Path(model_dir) / ACCEPTOR_CHECKPOINT, self.device)
        self.window_len = int(donor_ckpt["model_config"]["window_len"])
        if int(acceptor_ckpt["model_config"]["window_len"]) != self.window_len:
            raise ValueError(f"donor/acceptor window_len differ in {model_dir}")

    def model(self, kind: str):
        if kind == "donor":
            return self.donor
        if kind == "acceptor":
            return self.acceptor
        raise ValueError(f"Unsupported kind: {kind!r}")

    def logits(self, kind: str, windows: list[str]) -> list[float]:
        """Raw logits (sigmoid = P(true site)) for fixed-length windows."""
        if not windows:
            return []
        model = self.model(kind)
        encoded = np.stack([one_hot_encode_dna(w, self.window_len) for w in windows])
        out: list[float] = []
        with torch.inference_mode():
            for i in range(0, len(encoded), self.batch_size):
                tensor = torch.from_numpy(encoded[i : i + self.batch_size]).to(self.device)
                out.extend(model(tensor).view(-1).tolist())
        return out


def window_at_offset(raw_seq: str, kind: str, offset: int,
                     upstream: int = CNN_UPSTREAM, downstream: int = CNN_DOWNSTREAM) -> str:
    """Slice the CNN window out of an already 5'->3' oriented sequence.

    ``offset`` follows ``windowing.extract_site_window``: the index of the
    first intron base (G of GT) for a donor and of the first exonic base after
    the intron (2 nt past the A of AG) for an acceptor. Both models were
    trained with the dinucleotide starting at index ``upstream`` of the
    window, so the acceptor offset is moved back by 2 here. Sequence that runs
    off either end is N-padded, matching the N-padding the Markov code applied
    to truncated windows.
    """
    if kind == "acceptor":
        offset -= 2
    pad_left = max(0, upstream - offset)
    padded = ("N" * pad_left) + raw_seq.upper()
    offset += pad_left
    pad_right = max(0, (offset + downstream) - len(padded))
    padded = padded + ("N" * pad_right)
    window = reshape_site_sequence_4p(padded, kind, upstream, downstream, offset)
    if window is None:
        raise ValueError(f"cannot build a {kind} window at offset {offset}")
    return window
