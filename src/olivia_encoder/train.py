"""Train the Olivia encoder (defaults = the paper's checkpoint).

    python -m olivia_encoder.train --embedded-dir data/embedded/true_data_all_t37

K-fold training with AdamW; Head C is off for the first --pretrain-epochs.
Writes <results-dir>/encoder.pt and diagnostics.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset, TensorDataset

from .data import (
    DG_CACHE_FIELDS,
    NupackCache,
    SEED,
    SplitBundle,
    bundle_sequences,
    load_bundles,
    load_metadata,
)
from .targets import (
    DG_BIN_WIDTHS,
    DG_GRID_PAD_BINS,
    DG_HAS_NOT_FOUND,
    DG_LIMIT_QUANTILE,
    DG_STRUCTURE_TARGETS,
    HEAD_A_RULE_INDICES,
    HEAD_C_DG_INDICES,
    MIN_HAIRPIN_SPAN,
    PAIR_UNPAIRED,
    RULE_SPEC_DG,
    SEQUENCE_RULE_SPEC,
    bin_class_count,
    build_bin_targets,
    build_dg_bin_edges,
    build_pair_prob_targets,
    build_pairing_targets,
    build_structure_targets,
    calibrate_dg_limits,
    dg_grid_bounds,
    hard_pair_labels,
    hard_partner,
    pairing_baseline,
)
from .properties import (
    N_PROPERTIES,
    SEQ_PROPERTY_COUNT,
    TM_CONDITIONS,
    TM_DEGENERATE_MODES,
    extract_property_values,
    fit_binners,
    install_dg_properties,
    tm_summary,
    to_class_matrix,
)
from . import properties
from .density import (
    TermScaler,
    head_c_scores,
    head_c_terms,
    joint_density_loss,
    per_property_interaction_gain,
)
from .encoder import (
    HEAD_SET,
    PAIR_FEATURE_NAMES,
    PAIR_SCORE_FEATURE,
    PrimerEncoder,
    dg_center_tensors,
    dg_edge_tensors,
    head_a_score_dg,
    head_p_score,
    predicted_dg_classes,
    property_class_matrix,
)
from .synergy import (
    _HAIRPIN,
    plot_synergy_matrix,
    synergy_matrix,
)

# The repository root, two levels up from src/olivia_encoder/.
REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDED_DIR = REPO_DIR / "data" / "embedded" / "true_data_all_t37"
# Not results/encoder: that holds the checkpoint used in the paper.
DEFAULT_RESULTS_DIR = REPO_DIR / "results" / "encoder_retrained"
CHECKPOINT_NAME = "encoder.pt"


# Losses

# Class frequencies of the 3-class pairing target on the training corpus: unpaired
# 0.846, 5' 0.077, 3' 0.077.
PAIRING_BASELINE = 0.846


# Weight applied to the two paired classes relative to "unpaired".
PAIRING_PAIRED_WEIGHT = 1.0


# Width, in bins, of the Gaussian smoothing applied to the free-energy bin target.
BIN_SOFT_SIGMA = 1.0


def soft_bin_targets(
    cls: torch.Tensor, num_classes: int, ordinal_start: int = 1,
    sigma: float = BIN_SOFT_SIGMA,
) -> torch.Tensor:
    """(N,) class index -> (N, num_classes) soft distribution."""
    target = torch.zeros(cls.shape[0], num_classes, device=cls.device)
    if ordinal_start:
        is_not_found = cls < ordinal_start
        target[is_not_found, 0] = 1.0
        ordinal = ~is_not_found
    else:
        ordinal = torch.ones_like(cls, dtype=torch.bool)
    if ordinal.any():
        idx = torch.arange(ordinal_start, num_classes, device=cls.device, dtype=torch.float32)
        dist = idx.view(1, -1) - cls[ordinal].to(torch.float32).view(-1, 1)
        w = torch.exp(-0.5 * (dist / sigma) ** 2)
        target[ordinal, ordinal_start:] = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return target


def bin_loss(
    bin_logits: list[torch.Tensor], bin_targets: torch.Tensor,
    has_not_found: list[bool],
) -> torch.Tensor:
    """Mean soft-target cross-entropy over the structures, one term each."""
    losses = []
    for j, logits in enumerate(bin_logits):
        target = soft_bin_targets(bin_targets[:, j], logits.shape[1],
                                  ordinal_start=1 if has_not_found[j] else 0)
        losses.append(-(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean())
    return torch.stack(losses).mean()


def pairing_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Per-position pairing accuracy, and the paired-class recall that makes it
    interpretable.
    """
    model.eval()
    correct = total = 0
    paired_hit = paired_total = 0
    with torch.no_grad():
        for seq, pair_t, pair_v in loader:
            seq = seq.to(device).float()
            pair_t, pair_v = pair_t.to(device), pair_v.to(device)
            out = model(seq, apply_mask=False)
            pred = out["pair_logits"].argmax(dim=-1)
            correct += ((pred == pair_t) & pair_v).sum().item()
            total += pair_v.sum().item()
            is_paired = pair_v & (pair_t != PAIR_UNPAIRED)
            paired_hit += ((pred == pair_t) & is_paired).sum().item()
            paired_total += is_paired.sum().item()
    acc = correct / max(total, 1)
    return {
        "accuracy": acc,
        "paired_recall": paired_hit / max(paired_total, 1),
        "lift_over_baseline": acc - PAIRING_BASELINE,
    }


def soft_pairing_loss(pair_logits: torch.Tensor, pair_q: torch.Tensor,
                      pair_valid: torch.Tensor,
                      paired_weight: float = PAIRING_PAIRED_WEIGHT) -> torch.Tensor:
    """Per-position 3-way cross-entropy against a soft target, over real positions."""
    w_class = torch.tensor([1.0, paired_weight, paired_weight],
                           device=pair_logits.device, dtype=pair_logits.dtype)
    logp = F.log_softmax(pair_logits, dim=-1)
    per_pos = -(pair_q.to(logp.dtype) * w_class * logp).sum(dim=-1)
    w = pair_valid.to(logp.dtype)
    return (per_pos * w).sum() / w.sum().clamp_min(1e-8)


def stem_loss(pair_map_logits: torch.Tensor, top_idx: torch.Tensor,
              top_prob: torch.Tensor, unpaired: torch.Tensor,
              valid: torch.Tensor, paired_weight: float = 4.0) -> torch.Tensor:
    """Cross-entropy of the (L, L+1) stem head against the ensemble pair matrix."""
    logp = F.log_softmax(pair_map_logits, dim=-1)
    length = pair_map_logits.shape[1]

    gather_idx = top_idx.clamp_min(0).long()
    partner_logp = logp[..., :length].gather(-1, gather_idx)
    per_pos = -((top_prob.to(logp.dtype) * partner_logp).sum(dim=-1)
                + unpaired.to(logp.dtype) * logp[..., length])

    p_paired = top_prob.to(logp.dtype).sum(dim=-1)
    w = valid.to(logp.dtype) * (1.0 + (paired_weight - 1.0) * p_paired)
    return (per_pos * w).sum() / w.sum().clamp_min(1e-8)


@torch.no_grad()
def stem_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Hard-target accuracy of the stem head, for reporting."""
    model.eval()
    hit = tot = p_hit = p_tot = 0
    for seq, top_idx, top_prob, valid in loader:
        seq = seq.to(device).float()
        top_idx, top_prob, valid = top_idx.to(device), top_prob.to(device), valid.to(device)
        partner = hard_partner(top_idx, top_prob)
        logits = model(seq, apply_mask=False)["pair_map_logits"]
        pred = logits.argmax(dim=-1)
        tgt = torch.where(partner >= 0, partner, torch.full_like(partner, logits.shape[1]))
        hit += int(((pred == tgt) & valid).sum())
        tot += int(valid.sum())
        paired = valid & (partner >= 0)
        p_hit += int(((pred == tgt) & paired).sum())
        p_tot += int(paired.sum())
    return {"accuracy": hit / max(tot, 1), "partner_recall": p_hit / max(p_tot, 1),
            "paired_fraction": p_tot / max(tot, 1)}


# Training

LOSS_KEYS = ("total", "pair", "stem", "dg_bin", "density", "marg")


def compute_losses(model: PrimerEncoder, batch: tuple, device: torch.device,
                   weights: dict[str, float], centers: list[torch.Tensor],
                   edges: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    seq, pair_q, pair_v, top_idx, top_prob, unpaired, bin_t, seq_cls = [x.to(device) for x in batch]
    seq = seq.float()

    out = model(seq)
    l_pair = soft_pairing_loss(out["pair_logits"], pair_q, pair_v)
    l_stem = stem_loss(out["pair_map_logits"], top_idx, top_prob, unpaired, pair_v)
    # The same has_not_found flags as build_bin_targets, so both agree that
    # class 0 is an ordinary open-ended bin (the most stable one on the axis).
    l_bin = bin_loss(out["dg_bin_logits"], bin_t, DG_HAS_NOT_FOUND)
    # Head C's structural classes from Head A's current (detached) prediction.
    prop_t = predicted_dg_classes(seq_cls, out["dg_bin_logits"], centers, edges)
    l_den, l_marg = joint_density_loss(model.density, prop_t)

    total = (weights["pair"] * l_pair + weights["stem"] * l_stem
             + weights["dg_bin"] * l_bin + weights["density"] * (l_den + l_marg))
    parts = dict(zip(LOSS_KEYS, (total, l_pair, l_stem, l_bin, l_den, l_marg)))
    return total, {k: float(v) for k, v in parts.items()}


def evaluate_losses(model: PrimerEncoder, loader: DataLoader, device: torch.device,
                    weights: dict[str, float], centers: list[torch.Tensor],
                    edges: list[torch.Tensor]) -> dict[str, float]:
    model.eval()
    totals = {k: 0.0 for k in LOSS_KEYS}
    batches = 0
    devices = [device] if device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        torch.manual_seed(SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED)
        for batch in loader:
            _, parts = compute_losses(model, batch, device, weights, centers, edges)
            for k in LOSS_KEYS:
                totals[k] += parts[k]
            batches += 1
    return {k: v / max(batches, 1) for k, v in totals.items()}


def train_kfold(model_kwargs: dict, dataset: TensorDataset, device: torch.device,
                weights: dict[str, float], lr: float, k_folds: int, num_epochs: int,
                pretrain_epochs: int, batch_size: int, centers: list[torch.Tensor],
                edges: list[torch.Tensor]) -> dict[str, torch.Tensor]:

    k_folds = min(k_folds, len(dataset))
    if k_folds < 2:
        raise ValueError("Need at least 2 samples / folds.")

    # Pretraining: Head C off until Head A's predictions are meaningful.
    pretrain_weights = {
        "pair": weights["pair"],
        "stem": weights["stem"],
        "dg_bin": weights["dg_bin"],
        "density": 0.0,
    }

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    best_state, best_val = None, float("inf")

    for fold, (tr_idx, va_idx) in enumerate(kfold.split(np.arange(len(dataset))), start=1):
        print(f"\n=== Fold {fold}/{k_folds} ===")
        tr_loader = DataLoader(Subset(dataset, tr_idx.tolist()), batch_size=batch_size, shuffle=True)
        va_loader = DataLoader(Subset(dataset, va_idx.tolist()), batch_size=batch_size, shuffle=False)

        model = PrimerEncoder(**model_kwargs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        fold_state = copy.deepcopy(model.state_dict())
        fold_best = float("inf")

        for epoch in range(num_epochs):
            is_pretraining = epoch < pretrain_epochs
            active_weights = pretrain_weights if is_pretraining else weights
            phase_name = "Pretrain Head A" if is_pretraining else "Joint Tuning"

            model.train()
            running, batches = 0.0, 0
            for batch in tr_loader:
                loss, _ = compute_losses(model, batch, device, active_weights, centers, edges)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += loss.item()
                batches += 1

            val = evaluate_losses(model, va_loader, device, weights, centers, edges)
            print(f"  Epoch {epoch+1}/{num_epochs} [{phase_name}]: train={running/max(batches,1):.4f}  "
                  f"val={val['total']:.4f}  [pair={val['pair']:.4f} stem={val['stem']:.4f} "
                  f"dg_bin={val['dg_bin']:.4f} density={val['density']:.4f} "
                  f"marg={val['marg']:.4f}]")

            if val["total"] < fold_best:
                fold_best = val["total"]
                fold_state = copy.deepcopy(model.state_dict())

        if fold_best < best_val:
            best_val, best_state = fold_best, copy.deepcopy(fold_state)

    assert best_state is not None
    return best_state


# Pipeline
def make_tensors(bundle: SplitBundle, seq_len: int, cache: NupackCache):
    """All per-split training tensors: sequence, pairing and pair-map targets,
    NUPACK free energies, and the eight property values."""
    pair_q, pair_v = build_pairing_targets(bundle, seq_len, cache)
    top_idx, top_prob, unpaired = build_pair_prob_targets(bundle, seq_len, cache)
    dg_t, found_t = build_structure_targets(bundle, cache)
    prop_v, prop_f = extract_property_values(bundle, cache)
    return (bundle.sequence, pair_q, pair_v, top_idx, top_prob, unpaired,
            dg_t, found_t, prop_v, prop_f)


def run_training(bundles: dict[str, SplitBundle], device: torch.device, channels: int,
                 pair_channels: int, pair_layers: int,
                 density_dim: int, density_layers: int, density_heads: int,
                 weights: dict[str, float],
                 epochs: int, pretrain_epochs: int, learning_rate: float, k_folds: int,
                 results_dir: Path, batch_size: int, n_orders: int, cache: NupackCache,
                 dg_limits: list[float] | None = None,
                 dg_limit_quantile: float = DG_LIMIT_QUANTILE,
                 dg_grid_pad_bins: int = DG_GRID_PAD_BINS,
                 threshold_quantile: float = 0.95) -> pd.DataFrame:
    print(f"\n{'='*76}")
    print("  Olivia encoder: Head A (free energy), Head P (pair map), Head C (property density)")
    print("  Head C's structural coordinates are Head A's predicted dG")
    print("  Head C scores: C_marg (per-property) and C_inter (interaction)")
    print(f"  Head A regresses {len(DG_CACHE_FIELDS)} energies: {', '.join(DG_CACHE_FIELDS)}")
    print(f"  the noisy-OR rule fires on "
          f"{', '.join(DG_STRUCTURE_TARGETS[j] for j in HEAD_A_RULE_INDICES)}")
    print(f"  pretraining Head A for {pretrain_epochs} epochs before enabling Head C")
    print(f"  loss weights: {weights}")
    print(f"  NUPACK {cache.nupack_version} folds at {cache.fold_temp_c} degC, "
          f"Na+ {cache.sodium_m} M, Mg2+ {cache.magnesium_m} M")
    print(f"{'='*76}")

    seq_len = int(bundles["train"].sequence.shape[1])
    alphabet = int(bundles["train"].sequence.shape[2])

    # The limits define the dG property grids' anchors, so they have to be
    # settled before any property is binned.
    if dg_limits is None:
        dg_limits, calibration = calibrate_dg_limits(bundles["train"], cache, dg_limit_quantile)
        print(f"\n  dG limits at the {dg_limit_quantile:.2%} lower quantile of train:")
        for row in calibration:
            print(f"    {row['structure']:17s} {row['rule']:34s} "
                  f"-> dG <= {row['dg_limit']:+.3f} kcal/mol")
    else:
        calibration = [{"structure": n, "rule": "command line", "target_rate": float("nan"),
                        "dg_limit": v} for n, v in zip(DG_STRUCTURE_TARGETS, dg_limits)]
        print(f"\n  dG limits from the command line: {dg_limits}")
    for name, limit in zip(DG_STRUCTURE_TARGETS, dg_limits):
        RULE_SPEC_DG[f"{name}_min"] = limit

    # Grid spans from the training split's NUPACK energies.
    train_dg_raw = cache.energies_for(bundle_sequences(bundles["train"]), DG_CACHE_FIELDS)
    dg_bounds = dg_grid_bounds(train_dg_raw, dg_limits, DG_BIN_WIDTHS,
                               pad_bins=dg_grid_pad_bins)
    print(f"\n  dG grids span the full training range, padded {dg_grid_pad_bins} bins downward "
          f"(train split only -- no eval corpus is consulted):")
    for name, (lo, hi), width in zip(DG_STRUCTURE_TARGETS, dg_bounds, DG_BIN_WIDTHS):
        print(f"    {name:17s} [{lo:+7.2f}, {hi:+7.2f}] kcal/mol at width {width}")

    property_names = install_dg_properties(dg_limits, dg_bounds)

    print("  building NUPACK pairing / partner targets and the property vector ...")
    tensors = {s: make_tensors(bundles[s], seq_len, cache) for s in ("train", "val", "test")}

    bin_edges = build_dg_bin_edges(dg_bounds, dg_limits, DG_BIN_WIDTHS)
    bin_counts = [bin_class_count(e, hnf) for e, hnf in zip(bin_edges, DG_HAS_NOT_FOUND)]

    # Binner spans come from dg_bounds via the anchors.
    binners = fit_binners(tensors["train"][8], tensors["train"][9])
    property_class_counts = [b.n_classes for b in binners]
    dg_centers = dg_center_tensors(bin_edges, DG_BIN_WIDTHS, device)
    dg_edges = dg_edge_tensors(binners, device)

    print(f"  sequence: L={seq_len} A={alphabet} | property vector: D={N_PROPERTIES}")
    dg_train = tensors["train"][6].numpy()
    for j, name in enumerate(DG_STRUCTURE_TARGETS):
        v = dg_train[:, j]
        breach = float((v <= dg_limits[j]).mean())
        # The bottom open-ended class holds one representative dG.
        tail = float((v < bin_edges[j][0]).mean())
        print(f"    {name:17s} mean {v.mean():+7.3f}  min {v.min():+7.3f}  "
              f"max {v.max():+7.3f} kcal/mol | {bin_counts[j]:3d} bins, "
              f"{breach:.4f} at or below the limit, {tail:.4f} in the bottom class")

    tm_train = tensors["train"][8][:, property_names.index("tm")]
    tm_stats = tm_summary(bundle_sequences(bundles["train"]), tm_train)
    print(f"    nearest-neighbour Tm: mean {tm_stats['mean']:.2f} degC, "
          f"[{tm_stats['lo']:.2f}, {tm_stats['hi']:.2f}] | SantaLucia 1998 at "
          f"Na+ {TM_CONDITIONS['mv_conc_mm']:g} mM, Mg2+ {TM_CONDITIONS['dv_conc_mm']:g} mM, "
          f"dNTP {TM_CONDITIONS['dntp_conc_mm']:g} mM, oligo {TM_CONDITIONS['dna_conc_nm']:g} nM")
    print(f"      IUPAC codes '{properties.TM_DEGENERATE}': {tm_stats['n_degenerate']} of "
          f"{tm_stats['n']} primers are degenerate, mean Tm {tm_stats['mean_degenerate']:.2f} "
          f"degC against {tm_stats['mean_concrete']:.2f} for the concrete ones")

    pvalid = tensors["train"][2]
    baseline = pairing_baseline(tensors["train"][1], pvalid)
    pv = tensors["train"][1][pvalid]
    print(f"    3-class pairing mass: unpaired={float(pv[:, 0].mean()):.3f} "
          f"5'={float(pv[:, 1].mean()):.3f} 3'={float(pv[:, 2].mean()):.3f} "
          f"(constant-predictor accuracy {baseline:.3f})")

    # The ensemble target against the MFE pair list: how much of the corpus the MFE fold
    # is silent on.
    top_idx, top_prob = tensors["train"][3], tensors["train"][4]
    ens_paired = float(top_prob.sum(dim=-1)[pvalid].mean())
    confident = float((hard_partner(top_idx, top_prob)[pvalid] >= 0).float().mean())
    mfe_partner = torch.from_numpy(
        cache.partners_for(bundle_sequences(bundles["train"]), seq_len))
    mfe_paired = float((mfe_partner[pvalid] >= 0).float().mean())
    print(f"    stem target: E[paired] {ens_paired:.3f} per valid position, "
          f"{confident:.3f} with a partner above 0.5")
    print(f"      the MFE pair list pairs {mfe_paired:.3f} of them, and is "
          f"empty for {float((mfe_partner < 0).all(dim=1).float().mean()):.3f} of primers")

    def as_dataset(parts) -> TensorDataset:
        seq, pair_q, pair_v, top_idx, top_prob, unpaired, dg_t, found_t, prop_v, prop_f = parts
        # Only the six sequence classes are stored; the structural two are rebuilt per
        # batch.
        return TensorDataset(seq, pair_q, pair_v, top_idx, top_prob, unpaired,
                             build_bin_targets(dg_t, found_t, bin_edges, DG_HAS_NOT_FOUND),
                             to_class_matrix(prop_v, prop_f, binners)[:, :SEQ_PROPERTY_COUNT])

    datasets = {s: as_dataset(tensors[s]) for s in ("train", "val", "test")}

    model_kwargs = dict(
        seq_len=seq_len, alphabet=alphabet, bin_counts=bin_counts,
        property_class_counts=property_class_counts, channels=channels,
        pair_channels=pair_channels, pair_layers=pair_layers,
        density_dim=density_dim, density_layers=density_layers,
        density_heads=density_heads,
    )
    best_state = train_kfold(
        model_kwargs=model_kwargs, dataset=datasets["train"], device=device, weights=weights,
        lr=learning_rate, k_folds=k_folds, num_epochs=epochs, pretrain_epochs=pretrain_epochs,
        batch_size=batch_size, centers=dg_centers, edges=dg_edges,
    )

    model = PrimerEncoder(**model_kwargs).to(device)
    model.load_state_dict(best_state)

    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- scores
    def head_c_classes(split: str) -> torch.Tensor:
        """(N, 8) Head C input: six from the sequence, two from Head A."""
        seq_cls = to_class_matrix(tensors[split][8], tensors[split][9],
                                  binners)[:, :SEQ_PROPERTY_COUNT]
        return property_class_matrix(model, tensors[split][0], seq_cls, device,
                                     dg_centers, dg_edges)

    # Head C terms for every split; standardisation is fitted on train.
    prop_cls = {s: head_c_classes(s) for s in ("train", "val", "test")}
    c_terms = {s: head_c_terms(model.density, prop_cls[s], device)
               for s in ("train", "val", "test")}

    marg_scaler = TermScaler.fit(c_terms["train"]["s_marg"])
    inter_scaler = TermScaler.fit(c_terms["train"]["d"])
    print("\n  --- Head C term standardisation (fitted on train, frozen into the checkpoint) ---")
    print(f"    {'property':18s} {'s_marg mean':>12s} {'sd':>7s} | {'d mean':>9s} {'sd':>7s}")
    for j, name in enumerate(property_names):
        print(f"    {name:18s} {marg_scaler.mean[j]:12.3f} {marg_scaler.std[j]:7.3f} | "
              f"{inter_scaler.mean[j]:9.3f} {inter_scaler.std[j]:7.3f}")

    scores: dict[str, dict[str, np.ndarray]] = {}
    dg_pred: dict[str, np.ndarray] = {}
    pair_feats: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        seq_loader = DataLoader(TensorDataset(tensors[split][0]), batch_size=256, shuffle=False)
        a = head_a_score_dg(model, seq_loader, device, bin_edges, dg_limits, DG_BIN_WIDTHS)
        # `density` and `cls` are passed so that surprisal_joint -- and therefore
        # synergy -- comes back.
        c = head_c_scores(c_terms[split], marg_scaler, inter_scaler, n_orders=n_orders,
                          density=model.density, cls=prop_cls[split], device=device)
        p = head_p_score(model, seq_loader, device)
        dg_pred[split] = a["dg_pred"]
        pair_feats[split] = p["features"]
        scores[split] = {
            "a": a["score"], "p": p["score"], "a_structural": a["structural"],
            # The two decision scores, kept as separate heads.
            "c_marg": c["c_marg"], "c_inter": c["c_inter"],
            "c_marg_which": c["c_marg_which"], "c_inter_which": c["c_inter_which"],
            # Diagnostics, in nats, on the same sequence-free density.
            "c_surprisal_marg": c["surprisal_marg"],
            "c_surprisal_joint": c["surprisal_joint"],
            "c_synergy": c["synergy"],
        }

    # Pairing validation (soft targets collapsed to hard labels).
    struct_loader = DataLoader(
        TensorDataset(tensors["test"][0], hard_pair_labels(tensors["test"][1]),
                      tensors["test"][2]),
        batch_size=256, shuffle=False)
    pair_stats = pairing_accuracy(model, struct_loader, device)
    # pairing_accuracy computes its lift against the fixed PAIRING_BASELINE;
    # overwritten with the lift over the baseline measured on these targets.
    pair_stats["lift_over_baseline"] = pair_stats["accuracy"] - baseline
    partner_loader = DataLoader(
        TensorDataset(tensors["test"][0], tensors["test"][3], tensors["test"][4],
                      tensors["test"][2]),
        batch_size=256, shuffle=False)
    stem_stats = stem_accuracy(model, partner_loader, device)

    print(f"\n  3-class pairing (test): accuracy={pair_stats['accuracy']:.4f} "
          f"(constant-predictor {baseline:.3f}, lift {pair_stats['lift_over_baseline']:+.4f}) "
          f"paired_recall={pair_stats['paired_recall']:.4f}")
    print(f"  partner placement (test): accuracy={stem_stats['accuracy']:.4f} "
          f"partner_recall={stem_stats['partner_recall']:.4f} "
          f"(paired fraction {stem_stats['paired_fraction']:.4f})")

    # Head A calibration: predicted vs NUPACK dG.
    dg_truth = tensors["test"][6].numpy()
    dg_calib = []
    print("\n  --- Head A calibration: predicted dG vs NUPACK (test, kcal/mol) ---")
    for j, name in enumerate(DG_STRUCTURE_TARGETS):
        err = dg_pred["test"][:, j] - dg_truth[:, j]
        r = float(np.corrcoef(dg_pred["test"][:, j], dg_truth[:, j])[0, 1])
        dg_calib.append({"structure": name, "mae": float(np.abs(err).mean()),
                         "bias": float(err.mean()), "pearson_r": r})
        role = "rule + Head C" if j in HEAD_A_RULE_INDICES else "representation"
        print(f"    {name:17s} MAE {np.abs(err).mean():6.3f}  bias {err.mean():+6.3f}  "
              f"r {r:+.4f}  | NUPACK sd {dg_truth[:, j].std():6.3f}  ({role})")

    # Head P: pair-map feature spread on train.
    pf_train = pair_feats["train"]
    print("\n  --- Head P: pair-map features (train split) ---")
    for j, name in enumerate(PAIR_FEATURE_NAMES):
        col = pf_train[:, j]
        marker = "  <- Head P score" if name == PAIR_SCORE_FEATURE else ""
        # nan-aware: the span features are undefined for primers with no
        # confident pair, and how often that happens is itself worth seeing.
        missing = float(np.isnan(col).mean())
        note = f"  undefined {missing:.1%}" if missing else ""
        print(f"    {name:14s} mean {np.nanmean(col):7.3f}  sd {np.nanstd(col):7.3f}  "
              f"[p1 {np.nanpercentile(col, 1):7.3f}, p99 {np.nanpercentile(col, 99):7.3f}]"
              f"{note}{marker}")

    # How much of each coordinate the other seven properties explain:
    # gain_nats is MI(x_j ; x_{-j}).
    per_prop = per_property_interaction_gain(c_terms["test"], property_names)
    per_prop.to_csv(results_dir / "per_property_interaction_gain.csv", index=False)
    print("\n  --- what the other properties carry, per property (nats) ---")
    print(per_prop.round(4).to_string(index=False))
    print("    a near-zero row is a coordinate the density treats as independent of the")
    print("    rest, and therefore one C_inter can never fire on")

    # ---- Head C diagnostics
    ct = scores["test"]
    print("\n  --- Head C: is the joint structure real? (test split, nats) ---")
    print(f"    independence baseline    -sum_j log p(x_j)  = {ct['c_surprisal_marg'].mean():8.4f}")
    print(f"    joint                    -log p(x)          = {ct['c_surprisal_joint'].mean():8.4f}")
    print(f"    nats property-property interaction buys     = "
          f"{(ct['c_surprisal_marg'] - ct['c_surprisal_joint']).mean():8.4f}")

    syn = ct["c_synergy"]
    syn_pos_rate = float((syn > 0).mean())
    print(f"\n    worse-than-sum rate = {syn_pos_rate:.4f}   p99={np.percentile(syn, 99):+.4f}  "
          f"median={np.median(syn):+.4f}")

    # Which coordinate each Head C score fires on.
    print("\n  --- which coordinate each Head C score fires on (test split) ---")
    print(f"    {'property':18s} {'C_marg argmax':>14s} {'C_inter argmax':>15s}")
    for j, name in enumerate(property_names):
        print(f"    {name:18s} {float((ct['c_marg_which'] == j).mean()):13.1%} "
              f"{float((ct['c_inter_which'] == j).mean()):14.1%}")

    mi = synergy_matrix(model.density, device)
    off_diag = ~np.eye(N_PROPERTIES, dtype=bool)
    mi_asymmetry = float(np.abs(mi - mi.T)[off_diag].mean())
    mi_df = pd.DataFrame(mi, index=property_names, columns=property_names)
    print("\n  --- pairwise dependence learned (nats of mutual information) ---")
    print("      hairpin_dg / homodimer_dg rows are Head A's *predicted* free energy; "
          "read them against the calibration above")
    print(mi_df.round(4).to_string())
    mi_df.to_csv(results_dir / "property_dependence_matrix.csv")

    plot_synergy_matrix(mi, results_dir)

    thr = {h: float(np.quantile(scores["train"][h], threshold_quantile))
           for h in ("a", "c_marg", "c_inter", "p")}
    thr["a_structural"] = float(np.quantile(scores["train"]["a_structural"], threshold_quantile))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seq_len": seq_len, "alphabet": alphabet, "channels": channels,
            "head_set": HEAD_SET,
            "pair_channels": pair_channels, "pair_layers": pair_layers,
            "min_hairpin_span": MIN_HAIRPIN_SPAN,
            "density_dim": density_dim, "density_layers": density_layers,
            "density_heads": density_heads,
            "pretrain_epochs": pretrain_epochs,
            # What Head C's density is a distribution over.
            "head_c_density": "p(properties)",
            "head_c_conditioning": "none",
            # The frozen standardisation.
            "head_c_marg_scaler": marg_scaler.to_dict(),
            "head_c_inter_scaler": inter_scaler.to_dict(),
            "head_c_score_form": "max_j (term_j - mean_j) / std_j",
            "head_c_scores": ["c_marg", "c_inter"],
            # Where Head C's two structural coordinates come from: Head A's
            # posterior mean, so scoring needs no NUPACK cache.
            "head_c_dg_source": "head_a_expected_dg",
            "head_c_seq_property_count": SEQ_PROPERTY_COUNT,
            # Head P: the pair map read as a decision score.
            "pair_feature_names": list(PAIR_FEATURE_NAMES),
            "pair_score_feature": PAIR_SCORE_FEATURE,
            "dg_calibration_test": dg_calib,
            "structure_targets": DG_STRUCTURE_TARGETS,
            # Which of the four structures Head C carries and which the noisy-OR fires
            # on.
            "head_c_dg_indices": list(HEAD_C_DG_INDICES),
            "head_a_rule_indices": list(HEAD_A_RULE_INDICES),
            # How far each dG axis reaches, and that the span is the training
            # split's own range (no eval set chose these classes).
            "dg_grid_bounds": [list(b) for b in dg_bounds],
            "dg_grid_pad_bins": dg_grid_pad_bins,
            "dg_grid_source": "train_split_full_range",
            "rule_spec": dict(SEQUENCE_RULE_SPEC),
            "bin_edges": [e.tolist() for e in bin_edges], "bin_counts": bin_counts,
            "limits": dg_limits, "structure_has_not_found": DG_HAS_NOT_FOUND,
            # Everything needed to re-score this checkpoint without the cache, and
            # to prove which program and buffer produced its targets.
            "target_source": "nupack", "nupack_version": cache.nupack_version,
            "fold_temp_c": cache.fold_temp_c, "sodium_m": cache.sodium_m,
            "magnesium_m": cache.magnesium_m,
            "dg_cache_fields": list(DG_CACHE_FIELDS), "dg_bin_widths": DG_BIN_WIDTHS,
            # Tm model and buffer, saved with the weights.
            "tm_source": "nearest_neighbour_santalucia_1998",
            "tm_conditions": dict(TM_CONDITIONS),
            "tm_degenerate": properties.TM_DEGENERATE,
            "rule_spec_dg": dict(RULE_SPEC_DG), "dg_calibration": calibration,
            "dg_limit_quantile": dg_limit_quantile,
            "property_names": property_names,
            "property_edges": [b.edges.tolist() for b in binners],
            "property_class_counts": property_class_counts,
            "property_has_not_found": [s.has_not_found for s in properties.PROPERTIES],
            "n_orders": n_orders, "threshold_quantile": threshold_quantile,
            "threshold_head_a": thr["a"],
            "threshold_head_c_marg": thr["c_marg"],
            "threshold_head_c_inter": thr["c_inter"],
            "threshold_head_p": thr["p"],
            "threshold_head_a_structural": thr["a_structural"],
            "pairing_accuracy_test": pair_stats["accuracy"],
            "pairing_baseline": baseline,
            "pairing_paired_recall_test": pair_stats["paired_recall"],
            "partner_recall_test": stem_stats["partner_recall"],
            # Which pairing target trained the stem head.
            "pairing_target": "nupack_ensemble_pair_probabilities",
            "pair_top_k": cache.pair_top_k,
            "pair_prob_min": cache.pair_prob_min,
        },
        results_dir / CHECKPOINT_NAME,
    )

    row = {
        "model": "olivia_encoder",
        "pairing_accuracy_test": pair_stats["accuracy"],
        "pairing_baseline": baseline,
        "pairing_lift_over_baseline": pair_stats["lift_over_baseline"],
        "pairing_paired_recall_test": pair_stats["paired_recall"],
        "partner_accuracy_test": stem_stats["accuracy"],
        "partner_recall_test": stem_stats["partner_recall"],
        "partner_paired_fraction": stem_stats["paired_fraction"],
        # Fraction of the hairpin coordinate's surprisal the other seven
        # properties explain.
        "hairpin_interaction_gain_frac": float(
            per_prop.loc[per_prop["property"] == _HAIRPIN, "gain_frac"].iloc[0]),
        # One limit / MAE / r / bias set per structure, keyed by name.
        **{f"{c['structure']}_limit": lim for c, lim in zip(dg_calib, dg_limits)},
        **{f"{c['structure']}_pred_mae": c["mae"] for c in dg_calib},
        **{f"{c['structure']}_pred_r": c["pearson_r"] for c in dg_calib},
        **{f"{c['structure']}_pred_bias": c["bias"] for c in dg_calib},
        "threshold_head_a": thr["a"],
        "threshold_head_c_marg": thr["c_marg"],
        "threshold_head_c_inter": thr["c_inter"],
        "threshold_head_p": thr["p"],
        "threshold_head_a_structural": thr["a_structural"],
        "test_surprisal_marginal": float(ct["c_surprisal_marg"].mean()),
        "test_surprisal_joint": float(ct["c_surprisal_joint"].mean()),
        "test_nats_from_interaction": float(
            (ct["c_surprisal_marg"] - ct["c_surprisal_joint"]).mean()),
        # The two scores' own spread on held-out data.
        "test_c_marg_sd": float(ct["c_marg"].std()),
        "test_c_inter_sd": float(ct["c_inter"].std()),
        "test_synergy_median": float(np.median(syn)),
        "test_synergy_p99": float(np.percentile(syn, 99)),
        "test_worse_than_sum_rate": syn_pos_rate,
        "mean_pairwise_mutual_information": float(mi[off_diag].mean()),
        "max_pairwise_mutual_information": float(mi.max()),
        "mi_asymmetry": mi_asymmetry,
    }

    print("\n--- Summary ---")
    for k, v in row.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    return pd.DataFrame([row])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Olivia encoder: free-energy (A), pair-map (P) and "
                    "property-density (C) heads over a shared sequence encoder."
    )
    p.add_argument("--embedded-dir", type=Path, default=DEFAULT_EMBEDDED_DIR)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--nupack-cache", type=Path, default=None,
                   help="Folds from nupack_targets.py (default: <embedded-dir>/nupack/nupack_cache.npz).")
    p.add_argument("--dg-min", type=float, nargs=len(DG_CACHE_FIELDS), default=None,
                   metavar=tuple(f.upper() for f in DG_CACHE_FIELDS),
                   help=f"Violation limits in kcal/mol, one per structure in "
                        f"{DG_CACHE_FIELDS} order; a structure violates at or below its limit. "
                        f"All {len(DG_CACHE_FIELDS)} or none -- a mix of calibrated and "
                        f"hand-picked limits would make the noisy-OR combine two different "
                        f"notions of 'too stable'. Default: the --dg-limit-quantile lower "
                        f"quantile of the training corpus, per structure.")
    p.add_argument("--dg-limit-quantile", type=float, default=DG_LIMIT_QUANTILE,
                   help=f"Fraction of the training corpus each default dG limit rejects, i.e. how "
                        f"selective the rule is (default: {DG_LIMIT_QUANTILE * 100:.2f}%%). Ignored when "
                        f"--dg-min is given.")
    p.add_argument("--dg-grid-pad-bins", type=int, default=DG_GRID_PAD_BINS,
                   help=f"Bin widths of headroom below the training range of every dG grid "
                        f"(default: {DG_GRID_PAD_BINS}); the stable side only, since the top "
                        f"tail is a homogeneous 'barely folds' class and a free energy has a "
                        f"ceiling at 0 anyway. This is the only headroom there is: the grids "
                        f"span the training split and nothing else, deliberately, so there is "
                        f"no flag pointing them at an eval corpus.")
    # Buffer for the nearest-neighbour duplex Tm.
    p.add_argument("--tm-mv-conc", type=float, default=TM_CONDITIONS["mv_conc_mm"],
                   help="Monovalent cation concentration (mM) for the nearest-neighbour Tm.")
    p.add_argument("--tm-dv-conc", type=float, default=TM_CONDITIONS["dv_conc_mm"],
                   help="Divalent cation concentration (mM) for the nearest-neighbour Tm.")
    p.add_argument("--tm-dntp-conc", type=float, default=TM_CONDITIONS["dntp_conc_mm"],
                   help="dNTP concentration (mM); chelates divalent cations before they count.")
    p.add_argument("--tm-dna-conc", type=float, default=TM_CONDITIONS["dna_conc_nm"],
                   help="Total oligo strand concentration (nM) for the nearest-neighbour Tm.")
    p.add_argument("--tm-degenerate", choices=TM_DEGENERATE_MODES, default=properties.TM_DEGENERATE,
                   help="IUPAC codes: 'expected' averages the thermodynamics over the pool the "
                        "code denotes; 'first' scores only its first listed option "
                        "(default: expected).")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--pretrain-epochs", type=int, default=5,
                   help="Number of epochs to pretrain Head A before enabling Head C, whose "
                        "structural coordinates are Head A's prediction.")
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--k-folds", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--threshold-quantile", type=float, default=0.95)
    p.add_argument("--channels", type=int, default=96, help="Width of the dilated conv encoder.")
    p.add_argument("--pair-channels", type=int, default=32,
                   help="Channels in the (L, L) pair tensor.")
    p.add_argument("--pair-layers", type=int, default=3,
                   help="Residual 2D conv blocks over the pair tensor.")
    p.add_argument("--density-dim", type=int, default=96, help="Head C token width.")
    p.add_argument("--density-layers", type=int, default=2)
    p.add_argument("--density-heads", type=int, default=4)
    p.add_argument("--n-orders", type=int, default=4,
                   help="Autoregressive orders averaged when evaluating Head C's log-likelihood.")
    p.add_argument("--w-pair", type=float, default=1.0)
    p.add_argument("--w-stem", type=float, default=2.0,
                   help="Weight of the partner-placement loss.")
    p.add_argument("--w-dg-bin", type=float, default=2.0,
                   help="Weight of Head A's binned free-energy loss.")
    p.add_argument("--w-density", type=float, default=1.0,
                   help="Weight of Head C's joint density loss.")
    p.add_argument("--max-train-samples", type=int, default=0,
                   help="Subsample every split for quick smoke runs (0 = use all).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_metadata(args.embedded_dir)
    bundles = load_bundles(args.embedded_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Set before any property is extracted: the Tm computation reads these from
    # the properties module.
    TM_CONDITIONS.update(mv_conc_mm=args.tm_mv_conc, dv_conc_mm=args.tm_dv_conc,
                         dntp_conc_mm=args.tm_dntp_conc, dna_conc_nm=args.tm_dna_conc)
    properties.TM_DEGENERATE = args.tm_degenerate

    cache = NupackCache(args.nupack_cache or (args.embedded_dir / "nupack" / "nupack_cache.npz"))
    # All four limits or none.
    dg_limits = list(args.dg_min) if args.dg_min else None

    if args.max_train_samples:
        for split in bundles:
            b = bundles[split]
            n = min(args.max_train_samples, len(b.frame))
            b.sequence = b.sequence[:n]
            b.frame = b.frame.iloc[:n]
        print(f"[smoke run] truncated every split to <= {args.max_train_samples} samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    weights = {"pair": args.w_pair, "stem": args.w_stem, "dg_bin": args.w_dg_bin,
               "density": args.w_density}
    result = run_training(
        bundles=bundles, device=device, channels=args.channels,
        pair_channels=args.pair_channels, pair_layers=args.pair_layers,
        density_dim=args.density_dim, density_layers=args.density_layers,
        density_heads=args.density_heads,
        weights=weights,
        epochs=args.epochs, pretrain_epochs=args.pretrain_epochs,
        learning_rate=args.learning_rate, k_folds=args.k_folds,
        results_dir=args.results_dir, batch_size=args.batch_size, n_orders=args.n_orders,
        cache=cache, dg_limits=dg_limits, dg_limit_quantile=args.dg_limit_quantile,
        dg_grid_pad_bins=args.dg_grid_pad_bins,
        threshold_quantile=args.threshold_quantile,
    )
    print("\n=== Results ===")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
