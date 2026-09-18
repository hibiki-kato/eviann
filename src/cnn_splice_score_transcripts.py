#!/usr/bin/env python3
"""CNN replacement for score_transcripts_with_hmms.pl.

Usage: cnn_splice_score_transcripts.py TRANSCRIPTS.gff GENOME.fa MODEL_DIR [--device DEV]

Reads transcript/mRNA + exon records (gffread -F style attributes, ``ID=`` first),
scores every internal junction with the donor and acceptor CNNs and prints the
same seven space-separated columns the Perl scorer prints::

    id fix_score junction0 junction1 junction2 donor2 acceptor2

A CNN has no Markov order, so ``fix_score`` and the three junction columns all
carry the transcript score = min over junctions of (donor logit + acceptor
logit); ``donor2``/``acceptor2`` are the min donor/acceptor logits. Transcripts
without an internal junction keep the Perl default of 10000.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from cnn_splice.fasta import load_fasta  # noqa: E402
from cnn_splice.scoring import SiteModels  # noqa: E402
from cnn_splice.windowing import extract_site_window, junction_window_positions  # noqa: E402

NO_SCORE = 10000.0


def parse_transcripts(path: Path) -> list[tuple[str, list[list[str]]]]:
    transcripts: list[tuple[str, list[list[str]]]] = []
    transcript_id = ""
    exons: list[list[str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            if fields[2] in ("transcript", "mRNA"):
                if transcript_id:
                    transcripts.append((transcript_id, exons))
                exons = []
                transcript_id = fields[8].split(";")[0][3:]
            elif fields[2] == "exon" and transcript_id:
                exons.append(fields)
    if transcript_id:
        transcripts.append((transcript_id, exons))
    return transcripts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("transcripts", type=Path)
    parser.add_argument("genome", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk", type=int, default=2000, help="transcripts per scoring batch")
    args = parser.parse_args()

    genome = load_fasta(args.genome)
    models = SiteModels(args.model_dir, args.device)
    up, down = models.window_len // 2, models.window_len - models.window_len // 2
    transcripts = parse_transcripts(args.transcripts)
    out = sys.stdout

    for start in range(0, len(transcripts), args.chunk):
        chunk = transcripts[start : start + args.chunk]
        donor_windows: list[str] = []
        acceptor_windows: list[str] = []
        spans: list[tuple[int, int]] = []   # per transcript: [first, last) index into the window lists
        for tid, exons in chunk:
            first = len(donor_windows)
            for index in range(1, len(exons)):
                prev, cur = exons[index - 1], exons[index]
                seq = genome.get(cur[0])
                if seq is None:
                    sys.exit(f"Genome sequence {cur[0]} needed for transcript {tid} not found!")
                donor_pos, acceptor_pos = junction_window_positions(prev, cur, cur[6])
                d = extract_site_window(seq, donor_pos, cur[6], "donor", up, down)
                a = extract_site_window(seq, acceptor_pos, cur[6], "acceptor", up, down)
                if d is None or a is None:   # junction too close to a contig end
                    continue
                donor_windows.append(d)
                acceptor_windows.append(a)
            spans.append((first, len(donor_windows)))
        d_logits = models.logits("donor", donor_windows)
        a_logits = models.logits("acceptor", acceptor_windows)
        for (tid, _), (first, last) in zip(chunk, spans):
            if first == last:
                print(tid, NO_SCORE, NO_SCORE, NO_SCORE, NO_SCORE, NO_SCORE, NO_SCORE, file=out)
                continue
            d = d_logits[first:last]
            a = a_logits[first:last]
            combined = min(x + y for x, y in zip(d, a))
            print(tid, combined, combined, combined, combined, min(d), min(a), file=out)


if __name__ == "__main__":
    main()
