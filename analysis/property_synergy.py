"""Synergy surfaces, mutual information and conditional violation risk from the checkpoint.

    python analysis/property_synergy.py
    python analysis/property_synergy.py --top-k 10

Writes synergy_surfaces_top_mi, violation_risk_curves (paper figures) and
synergy_pair_summary.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from olivia_encoder import properties  # noqa: E402
from olivia_encoder.density import PropertyDensity, all_masked_logits  # noqa: E402
from olivia_encoder.encoder import build_model_from_checkpoint  # noqa: E402
from olivia_encoder.properties import (  # noqa: E402
    Binner,
    binners_from_checkpoint,
    install_dg_properties,
)
from olivia_encoder.synergy import (  # noqa: E402
    conditional_violation_risk,
    pairwise_synergy_surface,
    synergy_matrix,
)
from olivia_encoder.targets import (  # noqa: E402
    DG_STRUCTURE_TARGETS,
    HEAD_C_DG_INDICES,
    SEQUENCE_RULE_SPEC,
)

DEFAULT_RESULTS_DIR = REPO_DIR / "results" / "encoder"
CHECKPOINT_NAME = "encoder.pt"

# The two structural coordinates Head C carries, named exactly as dg_properties names
# them.
HAIRPIN, HOMODIMER = (DG_STRUCTURE_TARGETS[j] for j in HEAD_C_DG_INDICES)

# Filled by load_density, after install_dg_properties has rebound the property list.
PROPERTY_NAMES: list[str] = []
IDX: dict[str, int] = {}
N_PROPERTIES = 0


# Only targets with a violating class have a risk curve.
VIOLATION_PAIRS: list[tuple[str, str]] = [
    (HAIRPIN, "gc"), (HAIRPIN, "tm"), (HAIRPIN, "length"),
    (HAIRPIN, "complexity"), (HAIRPIN, "poly_x"), (HAIRPIN, "gc_clamp"),
    (HOMODIMER, "gc"), (HOMODIMER, "gc_clamp"), (HOMODIMER, "length"),
    ("tm", "gc"), ("tm", "length"),
    ("gc", "tm"), ("gc", "complexity"),
    ("poly_x", "complexity"), ("poly_x", "gc"),
]


# Loading

def load_density(checkpoint: Path, device: torch.device) -> tuple[PropertyDensity, list[Binner], dict]:
    """Head C and its discretisation, straight from the checkpoint."""
    global PROPERTY_NAMES, IDX, N_PROPERTIES

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"no checkpoint at {checkpoint}. Train one first:\n"
            f"    python -m olivia_encoder.train"
        )
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location="cpu")

    # build_model_from_checkpoint raises on a checkpoint of another head layout.
    model = build_model_from_checkpoint(ckpt)
    if ckpt.get("head_c_dg_source") != "head_a_expected_dg":
        raise ValueError(
            f"{checkpoint} has head_c_dg_source={ckpt.get('head_c_dg_source')!r}: its Head C reads "
            "structural coordinates from a NUPACK fold rather than from Head A, so the two "
            "structural rows of every figure here would mean something else."
        )

    names = install_dg_properties(list(ckpt["limits"]))
    PROPERTY_NAMES = names
    IDX = {n: i for i, n in enumerate(names)}
    N_PROPERTIES = len(names)
    if N_PROPERTIES != properties.N_PROPERTIES:
        raise ValueError(
            f"the dG property list has {N_PROPERTIES} entries but the density expects "
            f"{properties.N_PROPERTIES}."
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model.density, binners_from_checkpoint(ckpt), ckpt


# One surface, plus the probability mass behind each of its cells

@torch.no_grad()
def surface_with_mass(density: PropertyDensity, device: torch.device,
                      a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    """(K_a, K_b) synergy in nats and the joint mass p(a) p(b | a) of each cell."""
    syn = pairwise_synergy_surface(density, device, a, b)                     # (K_a, K_b)

    pa = torch.softmax(all_masked_logits(density, device)[a], dim=1).cpu().numpy()[0]
    ka = density.class_counts[a]
    cls = torch.zeros(ka, N_PROPERTIES, dtype=torch.long, device=device)
    cls[:, a] = torch.arange(ka, device=device)
    observed = torch.zeros_like(cls, dtype=torch.bool)
    observed[:, a] = True
    pb_a = torch.softmax(density(cls, observed)[b], dim=1).cpu().numpy()
    return syn, pa[:, None] * pb_a


def _boundaries(mask: np.ndarray) -> list[float]:
    """Class-edge coordinates where a boolean class mask changes value."""
    return [k - 0.5 for k in range(1, len(mask)) if mask[k] != mask[k - 1]]


def draw_surface(ax, syn: np.ndarray, mass: np.ndarray, a: int, b: int,
                 binners: list[Binner], mass_floor: float,
                 lim: float | None = None, n_ticks: int = 8,
                 label_axes: bool = True) -> tuple[object, float]:
    """One synergy panel. Returns the image and the colour limit used."""
    weak = mass < mass_floor
    shown = np.ma.masked_where(weak, syn)
    if lim is None:
        lim = float(np.abs(shown).max()) if shown.count() else 1.0
        lim = lim or 1.0

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#E8E8E8")
    im = ax.imshow(shown.T, cmap=cmap, vmin=-lim, vmax=lim, origin="lower", aspect="auto")

    ka, kb = syn.shape
    xt = np.linspace(0, ka - 1, min(ka, n_ticks)).astype(int)
    yt = np.linspace(0, kb - 1, min(kb, n_ticks)).astype(int)
    ax.set_xticks(xt); ax.set_xticklabels([binners[a].label(k) for k in xt], rotation=45, fontsize=6)
    ax.set_yticks(yt); ax.set_yticklabels([binners[b].label(k) for k in yt], fontsize=6)
    if label_axes:
        ax.set_xlabel(PROPERTY_NAMES[a], fontsize=8)
        ax.set_ylabel(PROPERTY_NAMES[b], fontsize=8)

    # The model's limits, as the class edge they fall on.
    for x in _boundaries(binners[a].violating_mask()):
        ax.axvline(x, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    for y in _boundaries(binners[b].violating_mask()):
        ax.axhline(y, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    # Separator for a "structure not found" class, if a binner has one (none of
    # the free-energy coordinates do).
    if binners[a].off:
        ax.axvline(0.5, color="#444444", linewidth=1.0)
    if binners[b].off:
        ax.axhline(0.5, color="#444444", linewidth=1.0)
    return im, lim


# Figures

# Manuscript figure style: 131 mm wide, 7-8 pt sans-serif, TrueType fonts.
MM = 1 / 25.4
NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.unicode_minus": False, "mathtext.default": "regular",
    "axes.facecolor": "white", "axes.edgecolor": "black", "axes.linewidth": 0.5,
    "axes.labelcolor": "black", "text.color": "black",
    "xtick.color": "black", "ytick.color": "black",
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "figure.facecolor": "white", "savefig.facecolor": "white",
}
# Axis labels with units, and the short form for the target property's y-axis label. Any
# property not listed falls back to its code name.
PRETTY = {
    "length": ("Length (nt)", "length"),
    "gc": ("GC content (%)", "GC"),
    "tm": ("T$_m$ (°C)", "T$_m$"),
    "complexity": ("Sequence complexity", "complexity"),
    "poly_x": ("Longest homopolymer (nt)", "poly-X"),
    "gc_clamp": ("GC clamp", "GC clamp"),
    "hairpin_dg": ("Hairpin ΔG (kcal/mol)", "hairpin ΔG"),
    "hairpin_dg_mfe": ("Hairpin ΔG, MFE (kcal/mol)", "hairpin ΔG"),
    "homodimer_dg": ("Homodimer ΔG (kcal/mol)", "homodimer ΔG"),
    "homodimer_dg_mfe": ("Homodimer ΔG, MFE (kcal/mol)", "homodimer ΔG"),
}


def _axis_label(name: str) -> str:
    return PRETTY.get(name, (name, name))[0]


def _short(name: str) -> str:
    return PRETTY.get(name, (name, name))[1]


def _wrapped_label(name: str) -> str:
    """_axis_label with a long one's unit on its own line, for the narrow 4-column panels."""
    label = _axis_label(name)
    return label.replace(" (", "\n(") if len(label) > 20 else label


def plot_top_mi(density: PropertyDensity, device: torch.device, binners: list[Binner],
                mi: np.ndarray, out_dir: Path, mass_floor: float, top_k: int) -> list[tuple[int, int]]:
    """The pairs the density itself ranks highest, not the ones we expected."""
    sym = 0.5 * (mi + mi.T)
    order = sorted(((sym[a, b], a, b) for a in range(N_PROPERTIES) for b in range(a + 1, N_PROPERTIES)),
                   reverse=True)
    chosen = [(a, b) for _, a, b in order[:top_k]]

    ncols = min(4, len(chosen))
    nrows = int(np.ceil(len(chosen) / ncols))
    surfaces = [surface_with_mass(density, device, a, b) for a, b in chosen]
    # One colour scale, so one colorbar: eight of them cost each panel a third of
    # its width, and the same red now means the same nats in every panel.
    lim = max(float(np.abs(syn[mass >= mass_floor]).max()) for syn, mass in surfaces)
    with plt.rc_context(NATURE_RC):
        # constrained_layout gives each column only the gutter its own labels
        # need; tight_layout gave all four the widest one, and the panels the rest.
        fig, axes = plt.subplots(nrows, ncols, figsize=(131 * MM, (42 * nrows + 8) * MM),
                                 squeeze=False, layout="constrained")
        # rect keeps the last column's titles, which are wider than their axes, on the page.
        fig.get_layout_engine().set(w_pad=0.5 * MM, h_pad=1 * MM, wspace=0.01, hspace=0.02,
                                    rect=(0, 0, 0.975, 1))
        for i, ax in enumerate(axes.ravel()):
            if i >= len(chosen):
                ax.axis("off")
                continue
            a, b = chosen[i]
            syn, mass = surfaces[i]
            im, _ = draw_surface(ax, syn, mass, a, b, binners, mass_floor, lim=lim, n_ticks=3,
                                 label_axes=False)
            # Print-size ticks and limit lines.
            ax.tick_params(labelsize=7)
            ax.set_xticklabels([f"{float(t.get_text()):.2g}" for t in ax.get_xticklabels()],
                               rotation=0, fontsize=7)
            ax.set_yticklabels([f"{float(t.get_text()):.2g}" for t in ax.get_yticklabels()],
                               fontsize=7)
            for ln in ax.lines:
                ln.set_linewidth(0.5)
            ax.set_xlabel(_wrapped_label(PROPERTY_NAMES[a]), labelpad=1)
            ax.set_ylabel(_wrapped_label(PROPERTY_NAMES[b]), labelpad=1)
        # Across the top, as the violation curves' legend is, so the panels keep the width.
        cb = fig.colorbar(im, ax=axes.ravel().tolist(), location="top", shrink=0.4, aspect=30,
                          pad=0.01)
        cb.locator = plt.MaxNLocator(5, symmetric=True, integer=True)
        cb.update_ticks()
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(labelsize=7, width=0.5, length=1.5)
        cb.set_label("Mutual Information (nats)", labelpad=2)
        for ext in ("png", "pdf"):
            out = out_dir / f"synergy_surfaces_top_mi.{ext}"
            fig.savefig(out, dpi=600)
            print(f"  Plot saved -> {out}")
        plt.close(fig)
    return chosen


def plot_violation_curves(density: PropertyDensity, device: torch.device,
                          binners: list[Binner], out_dir: Path, top_k: int = 8) -> None:
    """p(target breaks its limit | given = v) against the unconditional rate (top_k pairs).
    """
    pairs = [(t, g) for t, g in VIOLATION_PAIRS
             if t in IDX and g in IDX and binners[IDX[t]].violating_mask().any()]
    if not pairs:
        return
    curves = []
    for tname, gname in pairs:
        t, g = IDX[tname], IDX[gname]
        p, p0 = conditional_violation_risk(density, device, t, g)
        vmask = binners[t].violating_mask()
        risk = p[:, vmask].sum(axis=1)
        base = float(p0[vmask].sum())
        ratio = float((risk.max() + 1e-12) / (risk.min() + 1e-12))
        curves.append((ratio, tname, gname, risk, base))
    curves.sort(key=lambda c: c[0], reverse=True)
    curves = curves[:top_k]

    ncols = 4
    nrows = int(np.ceil(len(curves) / ncols))
    with plt.rc_context(NATURE_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(131 * MM, (45 * nrows + 5) * MM),
                                 squeeze=False, layout="constrained")
        # rect keeps the last column's titles, which are wider than their axes, on the page.
        fig.get_layout_engine().set(w_pad=0.5 * MM, h_pad=1 * MM, wspace=0.01, hspace=0.02,
                                    rect=(0, 0, 0.975, 1))
        for i, ax in enumerate(axes.ravel()):
            if i >= len(curves):
                ax.axis("off")
                continue
            ratio, tname, gname, risk, base = curves[i]
            g = IDX[gname]

            ks = np.arange(len(risk))
            for x in _boundaries(binners[g].violating_mask()):
                ax.axvline(x, color="#808080", linestyle=":", linewidth=0.5, zorder=1)
            ax.axhline(base, color="#0072B2", linestyle="--", linewidth=0.6, zorder=2)
            ax.plot(ks, risk, color="#D55E00", linewidth=0.9, zorder=3)
            # Two significant figures, as in the synergy panels, so three ticks fit.
            xt = np.linspace(0, len(risk) - 1, min(len(risk), 3)).astype(int)
            ax.set_xticks(xt)
            ax.set_xticklabels([f"{float(binners[g].label(k)):.2g}" for k in xt])
            ax.set_xlim(-0.5, len(risk) - 0.5)
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.set_xlabel(_wrapped_label(gname), labelpad=1)
            ax.set_ylabel(_short(tname)[0].upper() + _short(tname)[1:], labelpad=1)
            ax.set_title(f"max/min = {ratio:.0f}×", pad=4)

        handles = [Line2D([], [], color="#D55E00", linewidth=0.9, label="Conditional risk"),
                   Line2D([], [], color="#0072B2", linestyle="--", linewidth=0.6,
                          label="Unconditional rate"),
                   Line2D([], [], color="#808080", linestyle=":", linewidth=0.5,
                          label="Limit on conditioning property")]
        if len(curves) < nrows * ncols:
            # A free grid slot holds the legend.
            spare = axes.ravel()[len(curves)]
            spare.legend(handles=handles, loc="center", frameon=False,
                         handlelength=2.2, borderaxespad=0)
        else:
            fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False,
                       handlelength=2.2, columnspacing=1.2)
        for ext in ("png", "pdf"):
            out = out_dir / f"violation_risk_curves.{ext}"
            fig.savefig(out, dpi=600)
            print(f"  Plot saved -> {out}")
        plt.close(fig)


# The table behind the panels

def pair_summary(mi: np.ndarray, binners: list[Binner],
                 surfaces: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
                 mass_floor: float) -> pd.DataFrame:
    """One row per ordered pair."""
    rows = []
    for (a, b), (syn, mass) in surfaces.items():
        keep = mass >= mass_floor
        if not keep.any():
            continue
        w = mass * keep
        weighted = float((w * syn).sum() / w.sum())
        masked = np.where(keep, syn, -np.inf)
        ia, ib = np.unravel_index(int(np.argmax(masked)), syn.shape)

        syn_ba, mass_ba = surfaces[(b, a)]
        both = keep & (mass_ba.T >= mass_floor)
        cell_gap = float(np.abs(syn - syn_ba.T)[both].mean()) if both.any() else float("nan")

        rows.append({
            "a": PROPERTY_NAMES[a], "b": PROPERTY_NAMES[b],
            "mi_a_to_b": float(mi[a, b]), "mi_b_to_a": float(mi[b, a]),
            "mi_symmetrised": float(0.5 * (mi[a, b] + mi[b, a])),
            "mi_asymmetry": float(abs(mi[a, b] - mi[b, a])),
            "cell_asymmetry": cell_gap,
            "mean_synergy_massweighted": weighted,
            "max_synergy": float(syn[ia, ib]),
            "max_at_a": binners[a].label(int(ia)),
            "max_at_b": binners[b].label(int(ib)),
            "max_cell_mass": float(mass[ia, ib]),
            "cells_above_mass_floor": int(keep.sum()),
        })
    return pd.DataFrame(rows).sort_values("mi_symmetrised", ascending=False).reset_index(drop=True)


# Pipeline

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synergy, mutual information and conditional-risk diagnostics read from "
                    "the encoder's property density."
    )
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                   help="Directory holding encoder.pt.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Explicit checkpoint path; defaults to <results-dir>/encoder.pt.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where the plots go; defaults to --results-dir.")
    p.add_argument("--top-k", type=int, default=8,
                   help="How many pairs the model-ranked figure shows.")
    p.add_argument("--mass-floor", type=float, default=1e-5,
                   help="Cells with joint model mass below this are greyed out rather than allowed "
                        "to set the colour scale from a primer that does not exist.")
    p.add_argument("--violation-top-k", type=int, default=8,
                   help="How many pairs violation_risk_curves.png shows, ranked by max/min risk ratio.")
    p.add_argument("--device", type=str, default=None, help="cuda / cpu; default is cuda if present.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or (args.results_dir / CHECKPOINT_NAME)
    out_dir = args.output_dir or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*76}")
    print("  Head C pairwise synergy -- all property pairs")
    print(f"  checkpoint: {checkpoint}")
    print(f"  device: {device}   mass floor: {args.mass_floor:g}")
    print(f"{'='*76}")

    density, binners, ckpt = load_density(checkpoint, device)
    print(f"  properties: {', '.join(f'{n}[{b.n_classes}]' for n, b in zip(PROPERTY_NAMES, binners))}")
    print(f"  sequence limits: {SEQUENCE_RULE_SPEC}")
    for name, lim in zip(DG_STRUCTURE_TARGETS, ckpt["limits"]):
        carried = " (a Head C coordinate)" if name in IDX else ""
        print(f"  dG limit: {name:<18} violates at <= {lim:7.2f} kcal/mol{carried}")
    print(f"  the two structural coordinates are Head A's predicted dG, re-binned "
          f"({ckpt.get('head_c_dg_source')})")

    # Computed once and reused by every figure below; each is one forward pass
    # of K_a rows, so all 56 are cheaper than a single training step.
    surfaces = {(a, b): surface_with_mass(density, device, a, b)
                for a in range(N_PROPERTIES) for b in range(N_PROPERTIES) if a != b}

    mi = synergy_matrix(density, device)
    off = ~np.eye(N_PROPERTIES, dtype=bool)
    asym = float(np.abs(mi - mi.T)[off].mean())
    print(f"\n  mean MI = {mi[off].mean():.4f} nats   max = {mi.max():.4f}   "
          f"mean |MI(a,b) - MI(b,a)| = {asym:.4f}")
    if asym > 0.5 * mi[off].mean():
        print("    WARNING: the asymmetry is comparable to the MI itself. The two directions of a "
              "single quantity disagree, so read the surfaces as a fit in progress, not as biology; "
              "train longer before drawing conclusions from them.")

    summary = pair_summary(mi, binners, surfaces, args.mass_floor)
    csv = out_dir / "synergy_pair_summary.csv"
    summary.to_csv(csv, index=False)
    print(f"  Table saved -> {csv}")

    # One row per unordered pair for the console; the CSV keeps both directions.
    unique = summary[summary.apply(lambda r: IDX[r["a"]] < IDX[r["b"]], axis=1)]
    print("\n  --- strongest ten pairs ---")
    print(unique.head(10)[["a", "b", "mi_symmetrised", "mi_asymmetry", "cell_asymmetry",
                           "mean_synergy_massweighted", "max_synergy",
                           "max_at_a", "max_at_b"]].round(4).to_string(index=False))
    print("    cell_asymmetry is the error bar for a single square of a surface, and it is not "
          "mi_asymmetry:\n    averaging against p(a)p(b|a) cancels disagreements that reading a "
          "panel cell by cell does not.")
    print(f"    a row naming {HAIRPIN} or {HOMODIMER} is an interaction with Head A's prediction, "
          "not with a fold.")

    print()
    plot_top_mi(density, device, binners, mi, out_dir, args.mass_floor, args.top_k)
    plot_violation_curves(density, device, binners, out_dir, args.violation_top_k)

    print(f"\n  done -- {out_dir}")


if __name__ == "__main__":
    main()
