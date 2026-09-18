#!/usr/bin/env python3
"""Train the CNN v4.3 donor/acceptor splice-site models for eviann.sh.

Input site tables are the ones eviann.sh builds for the Markov matrices
(``don|acc|pair <tab> seqid <tab> pos <tab> strand ...``, 0-based ``pos``:
first intron base for ``don``, first exonic base after the intron for ``acc``).
Writes ``donor.pt`` and ``acceptor.pt`` into ``--model-dir``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from cnn_splice import ACCEPTOR_CHECKPOINT, CNN_DOWNSTREAM, CNN_UPSTREAM, DONOR_CHECKPOINT  # noqa: E402
from cnn_splice.model import train_from_args  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positive-sites", type=Path, required=True)
    parser.add_argument("--negative-sites", type=Path, required=True)
    parser.add_argument("--genome", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=int, default=CNN_UPSTREAM)
    parser.add_argument("--downstream", type=int, default=CNN_DOWNSTREAM)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=256)
    # DataLoader workers: one-hot encoding in the main process otherwise starves
    # the conv threads, which dominates wall time on CPU-only runs.
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--lr", type=float, default=0.0019023657363367618)
    parser.add_argument("--weight-decay", type=float, default=0.0032826322767654605)
    parser.add_argument("--eta-min-ratio", type=float, default=0.01)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001)
    parser.add_argument("--early-stop-min-delta-rel", type=float, default=0.0)
    parser.add_argument("--validation-metric", type=str, default="val_loss")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--conv-channels", type=str, default="128,256,512")
    parser.add_argument("--kernel-sizes", type=str, default="9,9,9")
    parser.add_argument("--block-dilations", type=str, default="1,6,12")
    parser.add_argument("--residual-channels", type=str, default="48,96,192")
    parser.add_argument("--head-type", type=str, default="gap")
    parser.add_argument("--fc-hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15561144760882448)
    parser.add_argument("--deformable-groups", type=int, default=4)
    parser.add_argument("--deformable-kernel-size", type=int, default=5)
    args = parser.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.donor_checkpoint = args.model_dir / DONOR_CHECKPOINT
    args.acceptor_checkpoint = args.model_dir / ACCEPTOR_CHECKPOINT
    train_from_args(args)


if __name__ == "__main__":
    main()
