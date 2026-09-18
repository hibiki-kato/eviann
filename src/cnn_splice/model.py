"""CNN v4.3 donor/acceptor splice-site classifiers for EviAnn.

Vendored from the eviann-nf-py ``cnn-v43`` branch (``eviann_py/cnn_model.py``),
itself adapted from IntronModel ``cnn_v3.py``/``cnn_v4.py``. Training reads the
``don``/``acc`` genome-coordinate site tables that eviann.sh already builds for
the Markov chain matrices, plus the genome FASTA, and writes one checkpoint per
task. ``load_task_model`` reloads them for scoring.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .arch import ArchConfig, OrganicSiteCNN
from .losses import build_focal_loss
from .runtime import (
    DNADataset,
    evaluate,
    normalize_checkpoint_state_dict,
    pick_device,
    stratified_split,
)
from .training_control import (
    get_metric_value,
    resolve_early_stopping_params,
    select_validation_score,
)
from .windowing import (
    acceptor_pos_from_tsv_fields,
    donor_pos_from_tsv_fields,
    extract_site_window,
)
from .fasta import load_fasta

CHECKPOINT_SITE_ARCH = "eviann_cnn_v4_3"


def load_task_examples(
    *,
    positive_sites: Path,
    negative_sites: Path,
    genome: dict[str, str],
    task: str,
    upstream: int,
    downstream: int,
) -> List[Tuple[str, int]]:
    """Read TSV rows for one task ("donor"/"acceptor"), windowed + labeled."""
    kind_field = "don" if task == "donor" else "acc"
    pos_reader = donor_pos_from_tsv_fields if task == "donor" else acceptor_pos_from_tsv_fields
    examples: List[Tuple[str, int]] = []
    for path, label in ((positive_sites, 1), (negative_sites, 0)):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                fields = line.rstrip("\n").split("\t")
                if fields[0] != kind_field:
                    continue
                seqid, pos, strand = pos_reader(fields)
                chrom_seq = genome.get(seqid)
                if chrom_seq is None:
                    raise KeyError(f"Genome sequence {seqid!r} not found")
                window = extract_site_window(chrom_seq, pos, strand, task, upstream, downstream)
                if window is None or "N" in window:
                    continue
                examples.append((window, label))
    return examples


def _build_epoch_loop(
    *,
    task: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: str,
    max_epochs: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    early_stop_min_delta_rel: float,
    validation_metric: str,
    grad_clip: float,
    checkpoint_path: Path,
    checkpoint_extra: Dict[str, object],
) -> Dict[str, object]:
    """Run the training epoch loop with the vendored early-stopping decision.

    Adapted from ``dev/IntronModel/src/models/cnn_v3.py:570-858``: the
    checkpoint-on-improvement rule, the ``patience_improved``/
    ``hit_val_loss_ceiling``/``patience_exhausted`` decision logic, and the
    stop-reason bookkeeping are the same decision logic as that function
    (AMP/grad-scaler/torch.compile/OOM-retry plumbing dropped).
    """
    best_score = -1e9
    best_metric_name = "acc@0.5"
    best_epoch = 0
    epoch_history: list[dict[str, object]] = []
    epochs_since_improvement = 0
    stopped_early = False
    stop_reason: Optional[str] = None
    initial_val_loss: Optional[float] = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += float(loss.detach().item())
        scheduler.step()
        train_loss = running_loss / max(1, len(train_loader))

        val_metrics = evaluate(model=model, loader=val_loader, device=device, loss_fn=criterion)
        val_loss = val_metrics.get("loss")

        score, score_name = select_validation_score(metrics=val_metrics, validation_metric=validation_metric)
        # best_score reflects the true global best regardless of the patience
        # rule below, so the saved checkpoint is always the single best
        # epoch seen so far. (cnn_v3.py:699-703)
        prev_best_score = best_score
        improved = score > (best_score + early_stop_min_delta)
        if improved:
            best_score = score
            best_metric_name = score_name
            best_epoch = epoch
            torch.save(
                {
                    "task": task,
                    "model_config": dict(checkpoint_extra),
                    "model_state": model.state_dict(),
                },
                checkpoint_path,
            )

        # Patience counting and the hard val_loss ceiling below are separate
        # from checkpoint selection above (cnn_v3.py:730-753): a val_loss dip
        # too small to clear the relative bar still isn't "progress" for
        # stopping purposes, but if it's the best epoch yet it is still
        # saved as the checkpoint.
        hit_val_loss_ceiling = False
        if score_name == "val_loss" and val_loss is not None:
            if epoch == 1:
                initial_val_loss = val_loss
            if early_stop_min_delta_rel > 0.0 and prev_best_score > -1e9:
                prev_best_val_loss = -prev_best_score
                required_val_loss = prev_best_val_loss * (1.0 - early_stop_min_delta_rel)
                patience_improved = val_loss < required_val_loss
            else:
                patience_improved = improved
            hit_val_loss_ceiling = (
                initial_val_loss is not None and epoch > 1 and val_loss >= initial_val_loss
            )
        else:
            patience_improved = improved

        if patience_improved:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "objective_metric": score_name,
                "objective_score": score,
                "improved": improved,
                "best_epoch": best_epoch,
            }
        )
        mark = "*" if improved else "-"
        val_loss_text = "nan" if val_loss is None else f"{val_loss:.4f}"
        print(
            f"[cnn_model:{task}] {mark} epoch {epoch}/{max_epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss_text} "
            f"score_metric={score_name} val_score={score:.4f} "
            f"best={best_score:.4f} (ep {best_epoch})",
            file=sys.stderr,
        )

        patience_exhausted = (
            early_stop_patience > 0 and epochs_since_improvement >= early_stop_patience
        )
        if patience_exhausted or hit_val_loss_ceiling:
            stopped_early = True
            stop_reason = "val_loss_ceiling" if hit_val_loss_ceiling else "patience"
            print(
                f"[cnn_model:{task}] early stop at epoch {epoch} "
                f"(reason={stop_reason}, patience={early_stop_patience}, "
                f"min_delta={early_stop_min_delta:g}, "
                f"min_delta_rel={early_stop_min_delta_rel:g})",
                file=sys.stderr,
            )
            break

    print(
        f"[cnn_model:{task}] done best_{best_metric_name}={best_score:.4f} at epoch {best_epoch}",
        file=sys.stderr,
    )
    return {
        "task": task,
        "best_metric": best_metric_name,
        "best_epoch": best_epoch,
        "best_score": float(best_score),
        "epoch_history": epoch_history,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
    }


def train_task(
    *,
    task: str,
    examples: List[Tuple[str, int]],
    window_len: int,
    checkpoint_path: Path,
    arch_config: ArchConfig,
    device: str,
    seed: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    eta_min_ratio: float,
    val_frac: float,
    grad_clip: float,
    max_epochs: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    early_stop_min_delta_rel: float,
    validation_metric: str,
    focal_gamma: float,
    num_workers: int,
) -> Dict[str, object]:
    """Train one donor or acceptor cnn_v4.3 model end to end."""
    n_pos = sum(label for _, label in examples)
    n_neg = len(examples) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Insufficient training examples for {task}: pos={n_pos}, neg={n_neg}.")

    train_ex, val_ex = stratified_split(examples, val_frac=val_frac, seed=seed)
    print(
        f"[cnn_model:{task}] device={device} total={len(examples)} "
        f"(pos={n_pos}, neg={n_neg}) train={len(train_ex)} val={len(val_ex)}",
        file=sys.stderr,
    )

    train_ds = DNADataset(train_ex, window_len=window_len)
    val_ds = DNADataset(val_ex, window_len=window_len)
    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_mem = device.startswith("cuda")
    train_loader = DataLoader(
        train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True, generator=generator,
        num_workers=num_workers, persistent_workers=(num_workers > 0), pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_ds, batch_size=min(batch_size, len(val_ds)), shuffle=False,
        num_workers=num_workers, persistent_workers=(num_workers > 0), pin_memory=pin_mem
    )

    train_pos = sum(label for _, label in train_ex)
    train_neg = len(train_ex) - train_pos
    criterion, loss_meta = build_focal_loss(train_pos=train_pos, train_neg=train_neg, focal_gamma=focal_gamma)
    criterion = criterion.to(device)

    torch.manual_seed(seed)
    model = OrganicSiteCNN(config=arch_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr * eta_min_ratio,
    )

    checkpoint_extra = {
        "site_arch": CHECKPOINT_SITE_ARCH,
        "window_len": window_len,
        "conv_channels": list(arch_config.conv_channels),
        "kernel_sizes": list(arch_config.kernel_sizes),
        "block_dilations": list(arch_config.block_dilations),
        "residual_channels": list(arch_config.residual_channels),
        "head_type": arch_config.head_type,
        "fc_hidden": arch_config.fc_hidden,
        "dropout": arch_config.dropout,
        "deformable_groups": arch_config.deformable_groups,
        "deformable_kernel_size": arch_config.deformable_kernel_size,
        "focal_gamma": loss_meta["focal_gamma"],
        "focal_alpha_pos": loss_meta["focal_alpha_pos"],
    }

    summary = _build_epoch_loop(
        task=task,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        max_epochs=max_epochs,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        early_stop_min_delta_rel=early_stop_min_delta_rel,
        validation_metric=validation_metric,
        grad_clip=grad_clip,
        checkpoint_path=checkpoint_path,
        checkpoint_extra=checkpoint_extra,
    )
    if not checkpoint_path.exists():
        # Guarantee an artifact even if no epoch ever registered as "improved"
        # (e.g. max_epochs=0 or a degenerate single-epoch run).
        torch.save(
            {"task": task, "model_config": dict(checkpoint_extra), "model_state": model.state_dict()},
            checkpoint_path,
        )
    return summary


def load_task_model(checkpoint_path: Path, device: str) -> Tuple[OrganicSiteCNN, dict]:
    """Load one eviann cnn_v4.3 checkpoint saved by ``train_task``."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Invalid checkpoint payload: {checkpoint_path}")
    config = ckpt.get("model_config")
    if not isinstance(config, dict) or config.get("site_arch") != CHECKPOINT_SITE_ARCH:
        raise ValueError(f"Not an eviann cnn_v4.3 checkpoint: {checkpoint_path}")
    arch_config = ArchConfig(
        conv_channels=config["conv_channels"],
        kernel_sizes=config["kernel_sizes"],
        block_dilations=config["block_dilations"],
        residual_channels=config["residual_channels"],
        head_type=config["head_type"],
        fc_hidden=config["fc_hidden"],
        dropout=config["dropout"],
        deformable_groups=config["deformable_groups"],
        deformable_kernel_size=config["deformable_kernel_size"],
    )
    model = OrganicSiteCNN(config=arch_config).to(device)
    model.load_state_dict(normalize_checkpoint_state_dict(ckpt["model_state"]))
    model.eval()
    return model, ckpt


def train_from_args(args: argparse.Namespace) -> None:
    """Train the donor and acceptor models from a parsed ``cnn_splice_train.py`` namespace."""
    device = pick_device(args.device)
    genome = load_fasta(args.genome)
    early_stop_patience, early_stop_min_delta, early_stop_min_delta_rel = resolve_early_stopping_params(
        args.early_stop_patience, args.early_stop_min_delta, args.early_stop_min_delta_rel,
    )
    window_len = args.upstream + args.downstream
    arch_config = ArchConfig(
        conv_channels=tuple(int(v) for v in args.conv_channels.split(",")),
        kernel_sizes=tuple(int(v) for v in args.kernel_sizes.split(",")),
        block_dilations=tuple(int(v) for v in args.block_dilations.split(",")),
        residual_channels=tuple(int(v) for v in args.residual_channels.split(",")),
        head_type=args.head_type,
        fc_hidden=args.fc_hidden,
        dropout=args.dropout,
        deformable_groups=args.deformable_groups,
        deformable_kernel_size=args.deformable_kernel_size,
    )

    for task, checkpoint_path in (("donor", args.donor_checkpoint), ("acceptor", args.acceptor_checkpoint)):
        examples = load_task_examples(
            positive_sites=args.positive_sites,
            negative_sites=args.negative_sites,
            genome=genome,
            task=task,
            upstream=args.upstream,
            downstream=args.downstream,
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        train_task(
            task=task,
            examples=examples,
            window_len=window_len,
            checkpoint_path=checkpoint_path,
            arch_config=arch_config,
            device=device,
            seed=args.seed,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eta_min_ratio=args.eta_min_ratio,
            val_frac=args.val_frac,
            grad_clip=args.grad_clip,
            max_epochs=args.max_epochs,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            early_stop_min_delta_rel=early_stop_min_delta_rel,
            validation_metric=args.validation_metric,
            focal_gamma=args.focal_gamma,
            num_workers=args.num_workers,
        )
