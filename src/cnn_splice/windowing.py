"""Genome-coordinate splice-site window extraction for the CNN v4.3 path.

VERIFIED FACT (see dev/IntronModel/run/.train_v43.sh): a splice-site input
window is 100 nt upstream + 100 nt downstream of the GT (donor) / AG
(acceptor) dinucleotide start position -- 200 nt total, offset-centered.
This is sliced by ``reshape_site_sequence_4p`` in
``dev/IntronModel/src/util/data_proc.py:855-892``, vendored verbatim below.

The genome-coordinate resolution (which raw genome slice to take, whether to
reverse-complement it for "-" strand, and where the splice-site offset falls
within that raw slice) is fresh code, written for this pipeline's own TSV
("don"/"acc" rows produced by ``eviann_py/splice_sites.py``, consumed via
``eviann_py/markov_model.py``'s ``_site_sequence`` position convention) and
GFF exon-field (as consumed by ``eviann_py/score_transcripts.py``'s
``MarkovSpliceScorer.junction_contexts``) coordinate conventions -- it does
NOT reuse markov's ``exon_bases``-anchored window math.
"""

from __future__ import annotations

from typing import Optional

from .fasta import revcomp

DONOR = "donor"
ACCEPTOR = "acceptor"


def reshape_site_sequence_4p(
    seq: str,
    site_type: str,
    upstream: Optional[int],
    downstream: Optional[int],
    splice_site_offset: int,
) -> Optional[str]:
    """Slice a splice-site sequence using independent upstream/downstream windows.

    Verbatim from ``dev/IntronModel/src/util/data_proc.py:855-892`` (the
    ``splice_site_offset is None`` auto-inference branch is dropped here
    since every caller in this module supplies an explicit offset).
    """
    if site_type not in {"donor", "acceptor"}:
        return None
    seq = seq.upper()
    start = splice_site_offset - upstream if upstream is not None else 0
    end = splice_site_offset + downstream if downstream is not None else len(seq)
    if start < 0 or end > len(seq) or start >= end:
        return None
    return seq[start:end]


def extract_site_window(
    chrom_seq: str,
    pos: int,
    strand: str,
    kind: str,
    upstream: int,
    downstream: int,
) -> Optional[str]:
    """Extract the offset-centered CNN window for one donor/acceptor site.

    Parameters
    ----------
    chrom_seq:
        Full (upper-cased) chromosome/scaffold sequence.
    pos:
        0-based genome-coordinate anchor, using the exact same convention as
        ``eviann_py/markov_model.py``'s ``_site_sequence``/
        ``eviann_py/splice_sites.py``'s legacy TSV rows: for "don" this is
        the forward-strand 0-based index of the first base of the intron
        (the G of GT on "+"; the position paired with strand-aware
        reverse-complementing on "-"); for "acc" it is the forward-strand
        0-based index of the first exonic base following the intron.
    strand:
        ``"+"`` or ``"-"``.
    kind:
        ``"donor"`` or ``"acceptor"``.
    upstream, downstream:
        Bases to keep before/after the splice-site dinucleotide start.

    Returns
    -------
    str | None
        The oriented (5'->3') window, or ``None`` if it would run off the
        contig, or ``pos``/``strand``/``kind`` are invalid.

    Notes
    -----
    Deriving the offset: extracting a strand-symmetric raw window
    ``chrom_seq[pos - margin : pos + margin]`` (reverse-complemented for
    "-" strand) places the dinucleotide's first base at a *fixed* local
    offset regardless of strand:

    - donor: offset == margin (both strands) -- the "pos" convention above
      makes ``pos`` itself the forward-strand index of G on "+", and one off
      of complement(G) on "-" once symmetric reverse-complementing is
      applied.
    - acceptor: offset == margin - 2 (both strands) -- "pos" marks the first
      exonic base, 2 nt downstream of the acceptor dinucleotide's first base
      (A of AG).

    This was derived by hand from ``eviann_py/markov_model.py``'s
    ``_site_sequence`` and ``eviann_py/splice_sites.py``'s
    ``_internal_junctions_from_tlf_exons``, and is checked empirically by
    the synthetic GT/AG placement test in this pipeline's verification
    script (see task report).
    """
    if kind not in (DONOR, ACCEPTOR):
        raise ValueError(f"Unsupported kind: {kind!r}")
    if strand not in ("+", "-"):
        raise ValueError(f"Unsupported strand: {strand!r}")

    margin = max(upstream, downstream) + 4
    start = pos - margin
    end = pos + margin
    if start < 0 or end > len(chrom_seq):
        return None
    raw = chrom_seq[start:end]
    if strand == "-":
        raw = revcomp(raw)
    offset = margin if kind == DONOR else margin - 2
    return reshape_site_sequence_4p(raw, kind, upstream, downstream, offset)


def donor_pos_from_tsv_fields(fields: list[str]) -> tuple[str, int, str]:
    """Return (seqid, pos, strand) for a "don" row of markov_*_introns.tsv."""
    return fields[1], int(fields[2]), fields[3]


def acceptor_pos_from_tsv_fields(fields: list[str]) -> tuple[str, int, str]:
    """Return (seqid, pos, strand) for an "acc" row of markov_*_introns.tsv."""
    return fields[1], int(fields[2]), fields[3]


def junction_window_positions(
    prev_exon: list[str], cur_exon: list[str], strand: str
) -> tuple[int, int]:
    """Return (donor_pos, acceptor_pos) in the TSV 0-based convention above,
    derived from a pair of consecutive GFF/GTF exon field rows (as produced
    by ``eviann_py/score_transcripts.py``'s ``parse_transcripts``).

    ``prev_exon``/``cur_exon`` are consecutive exons of one transcript in
    file order, i.e. increasing genomic coordinate (fields[3]=start,
    fields[4]=end, both 1-based-inclusive, matching standard GFF/GTF).
    """
    if strand == "+":
        donor_pos = int(prev_exon[4])
        acceptor_pos = int(cur_exon[3]) - 1
    else:
        donor_pos = int(cur_exon[3]) - 1
        acceptor_pos = int(prev_exon[4])
    return donor_pos, acceptor_pos
