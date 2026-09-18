"""CNN v4.3 splice-site model used by ``eviann.sh --splice-model cnn``.

Entry points live next to this package in ``src/``:

- ``cnn_splice_train.py``: positive/negative site tables + genome -> model dir
- ``cnn_splice_score_transcripts.py``: transcript GFF -> transcript_splice_scores.txt
- ``cnn_splice_score_sites.py``: line-oriented coprocess used by
  ``combine_gene_protein_gff.pl`` for CDS terminal-extension gating
"""

DONOR_CHECKPOINT = "donor.pt"
ACCEPTOR_CHECKPOINT = "acceptor.pt"
CNN_UPSTREAM = 100
CNN_DOWNSTREAM = 100
