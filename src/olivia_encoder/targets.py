"""Training targets from the NUPACK cache, and the free-energy bin grids.

Each dG grid is anchored so its limit is a bin edge; classes 0 and len(edges)
are open-ended tails. A structure violates when dG is at or below its limit.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .data import (
    DG_CACHE_FIELDS,
    NupackCache,
    SplitBundle,
    UNPAIRED_PARTNER,
    bundle_sequences,
)

PAIR_UNPAIRED, PAIR_5P, PAIR_3P = 0, 1, 2


def bin_class_count(edges: np.ndarray, has_not_found: bool = True) -> int:
    """Number of softmax classes for one structure: the interior bins between the
    edges, two open-ended tails, and a not-found class only where that outcome
    actually occurs."""
    return len(edges) + 1 + (1 if has_not_found else 0)


def build_bin_targets(
    values: torch.Tensor, found: torch.Tensor, edges_per_structure: list[np.ndarray],
    has_not_found: list[bool],
) -> torch.Tensor:
    """(N, S) int64 class index per structure, following the layout above."""
    out = np.zeros(values.shape, dtype=np.int64)
    tm_np, found_np = values.numpy(), found.numpy()
    for j, edges in enumerate(edges_per_structure):
        off = 1 if has_not_found[j] else 0
        bucket = np.digitize(tm_np[:, j], edges)   # 0 .. len(edges)
        cls = bucket + off
        if off:
            cls[found_np[:, j] <= 0.5] = 0         # not found overrides
        out[:, j] = cls
    return torch.from_numpy(out)


MIN_HAIRPIN_SPAN = 4


SEQUENCE_RULE_SPEC = {
    "gc_min": 20.0, "gc_max": 80.0,
    "tm_min": 55.0, "tm_max": 65.0,
    "length_min": 15, "length_max": 36,
    "poly_x_max": 5,
}


# Head A's structural targets are exactly the cached columns.
DG_STRUCTURE_TARGETS = list(DG_CACHE_FIELDS)


# Which of those four Head C carries as its two structural coordinates.
HEAD_C_DG_INDICES = (0, 3)


# Structures the noisy-OR rule uses (hairpin ensemble, homodimer MFE).
HEAD_A_RULE_INDICES = (0, 3)


# No 'not found' class for any free energy.
DG_HAS_NOT_FOUND = [False] * len(DG_CACHE_FIELDS)


# Bin widths per structure (kcal/mol).
DG_BIN_WIDTHS = [0.25, 0.25, 1.0, 1.0]


# Bins of headroom below the training range of each dG grid.
DG_GRID_PAD_BINS = 8


# Violation limits, in kcal/mol: a structure violates when its dG is <= the limit.
RULE_SPEC_DG = {f"{name}_min": None for name in DG_STRUCTURE_TARGETS}


# Default dG limits: the most stable 15% of training primers.
DG_LIMIT_QUANTILE = 0.15


def _renormalise(prob: np.ndarray, unpaired: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rescale a (..., K) / (...) pair of arrays so the K+1 support sums to 1."""
    total = prob.sum(axis=-1) + unpaired
    safe = np.where(total > 0.0, total, 1.0)
    return prob / safe[..., None], np.where(total > 0.0, unpaired / safe, 1.0)


def build_pair_prob_targets(bundle: SplitBundle, seq_len: int, cache: NupackCache
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Soft (L, L+1) pairing target for the stem head, as top-K + unpaired."""
    idx, prob, unpaired = cache.pair_probs_for(bundle_sequences(bundle), seq_len)
    pos = np.arange(seq_len)[None, :, None]
    too_close = (idx >= 0) & (np.abs(idx - pos) < MIN_HAIRPIN_SPAN)
    idx = np.where(too_close, -1, idx)
    prob = np.where(too_close, 0.0, prob)
    prob, unpaired = _renormalise(prob, unpaired)
    # int16 to save memory; widened to int64 per batch where gathered.
    return (torch.from_numpy(idx.astype(np.int16)),
            torch.from_numpy(prob.astype(np.float32)),
            torch.from_numpy(unpaired.astype(np.float32)))


def build_pairing_targets(bundle: SplitBundle, seq_len: int,
                          cache: NupackCache) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, L, 3) soft 3-class pairing target and (N, L) validity."""
    seqs = bundle_sequences(bundle)
    idx, prob, unpaired = cache.pair_probs_for(seqs, seq_len)
    prob, unpaired = _renormalise(prob, unpaired)

    pos = np.arange(seq_len)[None, :, None]
    q = np.zeros((len(seqs), seq_len, 3), dtype=np.float32)
    q[..., 0] = unpaired
    q[..., 1] = np.where(idx > pos, prob, 0.0).sum(axis=-1)   # 5' side of a stem
    q[..., 2] = np.where((idx >= 0) & (idx < pos), prob, 0.0).sum(axis=-1)

    valid = np.zeros((len(seqs), seq_len), dtype=bool)
    for i, s in enumerate(seqs):
        valid[i, :min(len(s), seq_len)] = True
    q[~valid] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return torch.from_numpy(q), torch.from_numpy(valid)


def pairing_baseline(pair_q: torch.Tensor, pair_v: torch.Tensor) -> float:
    """Accuracy of the best constant predictor on these soft 3-class targets."""
    q = pair_q[pair_v]
    if q.numel() == 0:
        return 0.0
    return float(q.reshape(-1, q.shape[-1]).mean(dim=0).max())


def hard_pair_labels(pair_q: torch.Tensor) -> torch.Tensor:
    """Most probable of the three classes, for accuracy reporting only."""
    return pair_q.argmax(dim=-1)


def hard_partner(idx: torch.Tensor, prob: torch.Tensor,
                 min_prob: float = 0.5) -> torch.Tensor:
    """Most probable partner per position, or -1, for accuracy reporting only."""
    best = prob.argmax(dim=-1, keepdim=True)
    top_p = prob.gather(-1, best).squeeze(-1)
    top_i = idx.long().gather(-1, best).squeeze(-1)
    return torch.where((top_p > min_prob) & (top_i >= 0), top_i,
                       torch.full_like(top_i, UNPAIRED_PARTNER))


def build_structure_targets(bundle: SplitBundle,
                            cache: NupackCache) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, 4) free energies and (N, 4) all-ones 'found' flags, in DG_CACHE_FIELDS order.
    """
    dg = cache.energies_for(bundle_sequences(bundle), DG_CACHE_FIELDS).astype(np.float32)
    return torch.from_numpy(dg), torch.ones_like(torch.from_numpy(dg))


# dG bins and thresholds (class layout in the module docstring)
def dg_grid_bounds(train_dg: np.ndarray, limits: list[float], widths: list[float],
                   pad_bins: int = DG_GRID_PAD_BINS) -> list[tuple[float, float]]:
    """(lo, hi) per structure spanning the *whole* training range, plus padding."""
    bounds = []
    for j, (limit, width) in enumerate(zip(limits, widths)):
        lo = float(train_dg[:, j].min()) - pad_bins * width
        hi = float(train_dg[:, j].max())
        lo = limit + width * math.floor((lo - limit) / width)
        hi = limit + width * math.ceil((hi - limit) / width)
        bounds.append((lo, hi))
    return bounds


def build_dg_bin_edges(bounds: list[tuple[float, float]], limits: list[float],
                       widths: list[float]) -> list[np.ndarray]:
    """Bin edges per structure, anchored so each limit lands exactly on an edge."""
    edges_per_structure = []
    for (lo, hi), limit, width in zip(bounds, limits, widths):
        k_lo = min(int(math.floor((lo - limit) / width + 1e-9)), 0)
        k_hi = max(int(math.ceil((hi - limit) / width - 1e-9)), 0)
        edges = limit + width * np.arange(k_lo, k_hi + 1, dtype=np.float64)
        edges_per_structure.append(edges.astype(np.float32))
    return edges_per_structure


def below_class_mask(edges: np.ndarray, limit: float) -> np.ndarray:
    """Boolean over classes: True where the whole class sits at or below `limit`."""
    n = bin_class_count(edges, False)
    mask = np.zeros(n, dtype=bool)
    for cls in range(n - 1):          # class k has upper edge edges[k]
        mask[cls] = edges[cls] <= limit
    return mask


def dg_bin_centers(edges: np.ndarray, width: float) -> np.ndarray:
    """Representative dG per class; the open-ended tails get a half-width beyond
    the grid."""
    centers = np.empty(bin_class_count(edges, False), dtype=np.float64)
    centers[0] = edges[0] - width / 2.0
    for cls in range(1, len(centers)):
        lower = edges[cls - 1]
        upper = edges[cls] if cls < len(edges) else lower + width
        centers[cls] = (lower + upper) / 2.0
    return centers


def calibrate_dg_limits(bundle: SplitBundle, cache: NupackCache,
                        quantile: float = DG_LIMIT_QUANTILE) -> tuple[list[float], list[dict]]:
    """dG limits set at a fixed lower quantile of the training corpus."""
    if not 0.0 < quantile < 0.5:
        raise ValueError(f"dG limit quantile must be in (0, 0.5), got {quantile}")
    dg = cache.energies_for(bundle_sequences(bundle), DG_CACHE_FIELDS)
    limits, report = [], []
    for j, name in enumerate(DG_STRUCTURE_TARGETS):
        limit = float(np.quantile(dg[:, j], quantile))
        limits.append(limit)
        report.append({"structure": name, "rule": f"most stable {quantile:.2%} of train",
                       "target_rate": quantile, "dg_limit": limit})
    return limits, report
