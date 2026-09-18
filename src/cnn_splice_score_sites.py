#!/usr/bin/env python3
"""Line-oriented splice-site scoring coprocess for combine_gene_protein_gff.pl.

Usage: cnn_splice_score_sites.py --model-dir DIR [--device DEV]

Protocol (one request per line on stdin, one response per line on stdout)::

    donor|acceptor <tab> SEQUENCE <tab> off1,off2,...   ->   logit1 <tab> logit2 ...

SEQUENCE is already oriented 5'->3'. An offset is the index of the first
intron base (the G of GT) for a donor and of the first exonic base after the
intron (2 past the A of AG) for an acceptor; windows that run off the sequence
are N-padded. A logit > 0 means the model calls the site genuine.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from cnn_splice.scoring import SiteModels, window_at_offset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    models = SiteModels(args.model_dir, args.device)
    up, down = models.window_len // 2, models.window_len - models.window_len // 2
    out = sys.stdout
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.rstrip("\n")
        if not line:
            out.write("\n")
            out.flush()
            continue
        kind, seq, offsets = line.split("\t", 2)
        offs = [int(o) for o in offsets.split(",") if o]
        windows = [window_at_offset(seq, kind, o, up, down) for o in offs]
        out.write("\t".join(f"{x:.4f}" for x in models.logits(kind, windows)) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
