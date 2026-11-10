"""Mutual information, synergy and conditional-risk diagnostics read from the density.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .targets import (
    DG_STRUCTURE_TARGETS,
    HEAD_C_DG_INDICES,
)
from .properties import N_PROPERTIES
from . import properties
from .density import (
    PropertyDensity,
    all_masked_logits,
)

@torch.no_grad()
def conditional_violation_risk(density: PropertyDensity, device: torch.device,
                               target: int, given: int) -> tuple[np.ndarray, np.ndarray]:
    """p(target in a violating class | given = each class), and with nothing given."""
    n_given = density.class_counts[given]
    cls = torch.zeros(n_given, N_PROPERTIES, dtype=torch.long, device=device)
    cls[:, given] = torch.arange(n_given, device=device)
    observed = torch.zeros(n_given, N_PROPERTIES, dtype=torch.bool, device=device)
    observed[:, given] = True

    logits = density(cls, observed)[target]
    p = torch.softmax(logits, dim=1).cpu().numpy()

    p0 = torch.softmax(all_masked_logits(density, device)[target], dim=1).cpu().numpy()[0]
    return p, p0


@torch.no_grad()
def pairwise_synergy_surface(density: PropertyDensity, device: torch.device,
                             a: int, b: int) -> np.ndarray:
    """(K_a, K_b) synergy = log p(b) - log p(b | a), in nats; positive = rarer than
    independence.
    """
    ka = density.class_counts[a]

    log_pb = F.log_softmax(all_masked_logits(density, device)[b], dim=1).cpu().numpy()[0]  # (K_b,)

    cls = torch.zeros(ka, N_PROPERTIES, dtype=torch.long, device=device)
    cls[:, a] = torch.arange(ka, device=device)
    observed = torch.zeros(ka, N_PROPERTIES, dtype=torch.bool, device=device)
    observed[:, a] = True
    log_pb_given_a = F.log_softmax(density(cls, observed)[b], dim=1).cpu().numpy()  # (K_a, K_b)

    return log_pb[None, :] - log_pb_given_a


@torch.no_grad()
def synergy_matrix(density: PropertyDensity, device: torch.device) -> np.ndarray:
    """(D, D) mutual information in nats between property pairs (entry (a, b) uses p(b |
    a)).
    """
    m = np.zeros((N_PROPERTIES, N_PROPERTIES), dtype=np.float64)
    for a in range(N_PROPERTIES):
        pa = torch.softmax(all_masked_logits(density, device)[a], dim=1).cpu().numpy()[0]
        for b in range(N_PROPERTIES):
            if a == b:
                continue
            syn = pairwise_synergy_surface(density, device, a, b)      # (K_a, K_b)
            cls = torch.zeros(density.class_counts[a], N_PROPERTIES, dtype=torch.long, device=device)
            cls[:, a] = torch.arange(density.class_counts[a], device=device)
            observed = torch.zeros_like(cls, dtype=torch.bool)
            observed[:, a] = True
            pb_a = torch.softmax(density(cls, observed)[b], dim=1).cpu().numpy()
            # MI = sum_a p(a) sum_b p(b|a) * PMI(a,b);  PMI = -synergy
            m[a, b] = float((pa[:, None] * pb_a * (-syn)).sum())
    return m


# Display names for the manuscript figure; any property not listed keeps its
# code name.
_MATRIX_LABELS = {
    "length": "Length", "gc": "GC content", "tm": "T$_m$",
    "complexity": "Complexity", "poly_x": "Homopolymer", "gc_clamp": "GC clamp",
    "hairpin_dg": "Hairpin ΔG", "hairpin_dg_mfe": "Hairpin ΔG (MFE)",
    "homodimer_dg": "Homodimer ΔG", "homodimer_dg_mfe": "Homodimer ΔG (MFE)",
}


def plot_synergy_matrix(mi: np.ndarray, results_dir: Path,
                        names: list[str] | None = None) -> None:
    """Pairwise MI heatmap (paper Supplementary Fig. "Learned dependence between
    primer properties"), 92 mm wide with 6-8 pt type. The diagonal is not a
    quantity and is drawn grey rather than as a zero."""
    names = list(names or properties.PROPERTY_NAMES)
    n = len(names)
    labels = [_MATRIX_LABELS.get(p, p) for p in names]
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "mathtext.default": "regular",
        "axes.edgecolor": "black", "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    }
    shown = np.ma.masked_where(np.eye(n, dtype=bool), mi)
    vmax = float(shown.max()) or 1.0
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#d9d9d9")
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(92 / 25.4, 84 / 25.4), layout="constrained")
        im = ax.imshow(shown, cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right",
                                                    rotation_mode="anchor")
        ax.set_yticks(range(n)); ax.set_yticklabels(labels)
        for i in range(n):
            for j in range(n):
                if i != j:
                    ax.text(j, i, f"{mi[i, j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if mi[i, j] > vmax * 0.55 else "black")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("Mutual information (nats)", fontsize=8)
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(labelsize=7, width=0.5, length=2)
        for ext in ("png", "pdf"):
            out = results_dir / f"property_dependence_matrix.{ext}"
            fig.savefig(out, dpi=600)
            print(f"  Plot saved -> {out}")
        plt.close(fig)


# Head C's two structural properties.
_HAIRPIN, _HOMODIMER = (DG_STRUCTURE_TARGETS[j] for j in HEAD_C_DG_INDICES)


