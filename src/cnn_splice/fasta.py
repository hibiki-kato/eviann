"""Minimal FASTA reader and reverse complement (no Biopython dependency)."""

from __future__ import annotations

from pathlib import Path

RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def load_fasta(path: Path) -> dict[str, str]:
    """Return {seqid: upper-case sequence}; seqid = header up to the first space."""
    seqs: dict[str, list[str]] = {}
    name = ""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = []
            elif name:
                seqs[name].append(line)
    return {key: "".join(value).upper() for key, value in seqs.items()}


def revcomp(seq: str) -> str:
    return seq.translate(RC)[::-1].upper()
