"""Embedding separation of accepted vs rejected primers (paper Supplementary embedding figure).

    python analysis/embedding_separation.py --eval-set test-HIV1 --eval-root <dir>

Three representations of the same primers -- raw one-hot, the untrained encoder
and the trained encoder -- each with a cross-validated linear-probe AUC and a
t-SNE layout. Needs the embedded primer3-labelled evaluation corpora.
Writes embedding_separation_top.{png,pdf,svg,npz}.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from olivia_encoder.data import SEED, load_bundles  # noqa: E402
from olivia_encoder.encoder import build_model_from_checkpoint  # noqa: E402

DEFAULT_RESULTS_DIR = REPO_DIR / "results" / "encoder"
CHECKPOINT_NAME = "encoder.pt"

# Where the primer3-labelled evaluation corpora (built with build_dataset
# --reference-dir) live; --eval-root overrides it.
DEFAULT_EVAL_ROOT = REPO_DIR / "data" / "embedded" / "correct_primer3_web"

# name -> (rejected dir, matching accepted dir) under the eval root.
EVAL_PAIRS: dict[str, tuple[str, str]] = {
    name: (f"{name}_bad_primers_embedded", f"{name}_variable_tm_primers_embedded")
    for name in ("test-SARS", "test-HIV1", "test-Phage")
}

# Okabe-Ito blue / orange (colour-blind and greyscale safe).
CLASS_COLORS = {0: "#0072B2", 1: "#E69F00"}


# Data

def pad_sequence_length(seq: torch.Tensor, target_len: int) -> torch.Tensor:
    """Zero-pad an (N, L, A) one-hot up to the model's fixed length."""
    cur = seq.shape[1]
    if cur == target_len:
        return seq
    if cur > target_len:
        raise ValueError(f"sequence length {cur} exceeds the checkpoint's {target_len}")
    pad = torch.zeros((seq.shape[0], target_len - cur, seq.shape[2]), dtype=seq.dtype)
    return torch.cat([seq, pad], dim=1)


def stack_dir(directory: Path, seq_len: int) -> torch.Tensor:
    """One-hot sequences of all three splits of one eval directory, concatenated."""
    bundles = load_bundles(directory)
    return torch.cat([pad_sequence_length(bundles[s].sequence, seq_len)
                      for s in ("train", "val", "test")], dim=0)


def load_eval_set(name: str, seq_len: int, max_per_class: int, seed: int,
                  eval_root: Path) -> tuple[torch.Tensor, np.ndarray]:
    """(sequences, labels): equal numbers of accepted (0) and rejected (1) primers."""
    bad_dir, good_dir = (eval_root / d for d in EVAL_PAIRS[name])
    for d in (bad_dir, good_dir):
        if not d.exists():
            raise FileNotFoundError(f"eval directory missing: {d}")
    bad, good = stack_dir(bad_dir, seq_len), stack_dir(good_dir, seq_len)
    rng = np.random.default_rng(seed)

    def take(n: int, k: int) -> np.ndarray:
        return np.arange(n) if k <= 0 or k >= n else np.sort(rng.choice(n, k, replace=False))

    k = max_per_class if max_per_class > 0 else min(len(bad), len(good))
    k = min(k, len(bad), len(good))
    ib, ig = take(len(bad), k), take(len(good), k)
    seq = torch.cat([good[ig], bad[ib]], dim=0)
    y = np.concatenate([np.zeros(len(ig), dtype=int), np.ones(len(ib), dtype=int)])
    return seq, y


def load_model(checkpoint: Path, device: torch.device,
               trained: bool) -> torch.nn.Module:
    if not checkpoint.exists():
        raise FileNotFoundError(f"no checkpoint at {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint(ckpt)
    if trained:
        model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt


# Representations

@torch.no_grad()
def pooled_embedding(model: torch.nn.Module, seq: torch.Tensor, device: torch.device,
                     batch_size: int = 256) -> np.ndarray:
    """`pooled` from the forward pass: the representation Head A's dG bins read."""
    out = []
    for (batch,) in DataLoader(TensorDataset(seq), batch_size=batch_size, shuffle=False):
        out.append(model(batch.to(device).float())["pooled"].cpu().numpy())
    return np.concatenate(out, axis=0)


def separation_metrics(X: np.ndarray, y: np.ndarray, seed: int,
                       n_splits: int = 5, sil_max: int = 4000) -> dict[str, float]:
    """Cross-validated logistic-probe AUC and silhouette on the full-dimensional vectors."""
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    p = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, p))

    Z = StandardScaler().fit_transform(X)
    if len(Z) > sil_max:
        idx = np.random.default_rng(seed).choice(len(Z), sil_max, replace=False)
        Z, ys = Z[idx], y[idx]
    else:
        ys = y
    return {"probe_auc": auc, "silhouette": float(silhouette_score(Z, ys)),
            "n": int(len(X)), "dim": int(X.shape[1])}


def project_2d(X: np.ndarray, seed: int, perplexity: float, pca_dim: int = 50) -> np.ndarray:
    """PCA to `pca_dim`, then t-SNE."""
    Z = StandardScaler().fit_transform(X)
    if Z.shape[1] > pca_dim:
        Z = PCA(n_components=pca_dim, random_state=seed).fit_transform(Z)
    return TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=seed, max_iter=1000).fit_transform(Z)


# Figure

def display_subsample(y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """Indices of at most `per_class` points per class, sorted."""
    if per_class <= 0:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    keep = []
    for lab in np.unique(y):
        idx = np.flatnonzero(y == lab)
        keep.append(idx if len(idx) <= per_class
                    else rng.choice(idx, size=per_class, replace=False))
    return np.sort(np.concatenate(keep))


def plot_embedding_top(panels: list[tuple[str, np.ndarray, dict[str, float]]],
                       y: np.ndarray, sel: np.ndarray, out_dir: Path) -> None:
    """One t-SNE panel per representation, manuscript style (131 mm wide)."""
    mm = 1 / 25.4
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans",
                            "Nimbus Sans", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.facecolor": "white", "axes.edgecolor": "black", "axes.linewidth": 0.5,
        "text.color": "black", "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, len(panels), figsize=(131 * mm, 58 * mm))
        for ax, (title, xy, met) in zip(axes, panels):
            for lab in (0, 1):
                m = sel & (y == lab)
                ax.scatter(xy[m, 0], xy[m, 1], s=1.5, alpha=0.8,
                           c=CLASS_COLORS[lab], linewidths=0, rasterized=True)
            name = title.split(" -- ")[0]
            name = name[0].upper() + name[1:].lower()
            ax.set_title(f"{name}\nprobe AUC = {met['probe_auc']:.3f}", pad=3)
            ax.set_xticks([]); ax.set_yticks([])
        axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")
        handles = [Line2D([], [], marker="o", linestyle="", markersize=4,
                          markeredgewidth=0, color=CLASS_COLORS[lab], label=name)
                   for lab, name in ((0, "Accepted"), (1, "Rejected"))]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 1.0), handletextpad=0.2,
                   columnspacing=1.5, borderaxespad=0)
        fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.0)
        # Points are rasterized in the vector copies; text and frames stay vector.
        for ext in ("png", "pdf", "svg"):
            out = out_dir / f"embedding_separation_top.{ext}"
            fig.savefig(out, dpi=600)
            print(f"  Plot saved -> {out}")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embedding separation of accepted vs rejected primers for the trained encoder."
    )
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Explicit checkpoint path; defaults to <results-dir>/encoder.pt.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT,
                   help="Directory holding the embedded primer3-labelled evaluation corpora.")
    p.add_argument("--eval-set", type=str, default="test-HIV1", choices=sorted(EVAL_PAIRS))
    p.add_argument("--max-per-class", type=int, default=1500,
                   help="Primers per class (0 = all).")
    p.add_argument("--plot-per-class", type=int, default=500,
                   help="Points per class drawn (0 = all); metrics use every primer.")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or (args.results_dir / CHECKPOINT_NAME)
    out_dir = args.output_dir or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, ckpt = load_model(checkpoint, device, trained=True)
    seq, y = load_eval_set(args.eval_set, ckpt["seq_len"], args.max_per_class, args.seed,
                           args.eval_root)
    print(f"  {args.eval_set}: {int((y == 0).sum())} accepted + {int((y == 1).sum())} rejected")

    random_model, _ = load_model(checkpoint, device, trained=False)
    reps = [
        ("raw one-hot sequence", seq.reshape(len(y), -1).numpy()),
        ("pooled, RANDOM INIT", pooled_embedding(random_model, seq, device)),
        ("pooled, TRAINED", pooled_embedding(model, seq, device)),
    ]
    panels = []
    for title, X in reps:
        met = separation_metrics(X, y, args.seed)
        print(f"    {title:<22} dim={met['dim']:<5} probe AUC={met['probe_auc']:.4f}  "
              f"silhouette={met['silhouette']:+.4f}")
        panels.append((title, project_2d(X, args.seed, args.perplexity), met))

    sel = np.zeros(len(y), dtype=bool)
    sel[display_subsample(y, args.plot_per_class, args.seed)] = True
    np.savez(out_dir / "embedding_separation_top.npz", y=y, sel=sel,
             titles=np.array([t for t, _, _ in panels]),
             xy=np.stack([xy for _, xy, _ in panels]),
             probe_auc=np.array([met["probe_auc"] for _, _, met in panels]),
             dim=np.array([met["dim"] for _, _, met in panels]))
    plot_embedding_top(panels, y, sel, out_dir)


if __name__ == "__main__":
    main()
