# EviAnn
An evidence-based eukaryotic annotation pipeline.  This is the eviann submodule.  For the EviAnn releases please go to EviAnn_release repository.

## CNN splice-site model (`--splice-model cnn`)

`eviann.sh --splice-model cnn` replaces the Markov chain / WAM splice-site
scores with a donor and an acceptor CNN (the IntronModel cnn_v4.3 architecture,
PyTorch) trained on the same positive (protein-supported introns) and negative
(weak RNA-seq junctions) sites the Markov matrices are built from. The CNN is
used everywhere the WAM was: transcript splice filtering, protein-only locus
filtering, the `--untrusted-cds` filter, and the CDS terminal-extension checks
in `combine_gene_protein_gff.pl` (which talks to `cnn_splice_score_sites.py`
as a coprocess; a site is called genuine when its logit is > 0).

Files: `src/cnn_splice/` (model, training loop, windowing),
`src/cnn_splice_train.py`, `src/cnn_splice_score_transcripts.py`,
`src/cnn_splice_score_sites.py`. Requires `torch` and `numpy` in the Python
that `python3` resolves to; a GPU is used when available. Training runs once per
run directory (`<genome>.cnn_splice/`) and is reused by `-c` reruns; delete the
directory to retrain. Extra training options (e.g. `--device cpu`,
`--max-epochs 20`) can be passed through the `CNN_SPLICE_TRAIN_ARGS`
environment variable. RNA-seq evidence is required (negative sites).
