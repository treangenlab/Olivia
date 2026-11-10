"""The Olivia encoder: a shared sequence encoder with three scoring heads.

    Head A (paper M_dG)  S_A     noisy-OR of p(dG <= limit) from predicted
    free-energy bins
    Head P (paper M_Pi)  S_P     largest per-position pairing probability in the
    pair map
    Head C (paper M_C)   C_marg, C_inter   property density (density.py)

A 1D conv stem with four dilated residual blocks feeds a pairwise (L, L) stem.
Head C reads six sequence properties plus Head A's predicted free energies.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .sequence import (
    BASES,
)
from .targets import (
    HEAD_A_RULE_INDICES,
    HEAD_C_DG_INDICES,
    MIN_HAIRPIN_SPAN,
    below_class_mask,
    dg_bin_centers,
)
from .properties import (
    Binner,
    SEQ_PROPERTY_COUNT,
)
from .density import (
    PropertyDensity,
    TermScaler,
)

class DilatedBlock(nn.Module):
    """Residual dilated conv. Stacking dilations 1,2,4,8 with kernel 5 reaches a
    receptive field of ~61 positions, which spans the whole 40-base primer."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=5,
                              padding=2 * dilation, dilation=dilation)
        # GroupNorm needs channels divisible by the group count; fall back to
        # whatever divides so --channels stays free to tune.
        groups = next((g for g in (8, 4, 2, 1) if channels % g == 0), 1)
        self.norm = nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.norm(self.conv(x)))


class SequenceEncoder(nn.Module):
    """(N, L, A) one-hot -> (N, C, L) per-position features."""

    def __init__(self, alphabet: int, channels: int = 96, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.stem = nn.Conv1d(alphabet, channels, kernel_size=5, padding=2)
        self.blocks = nn.ModuleList([DilatedBlock(channels, d) for d in dilations])
        self.channels = channels

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.stem(seq.transpose(1, 2)))      # (N, C, L)
        for block in self.blocks:
            h = block(h)
        return h


def masked_pool(h: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """(N, C, L) -> (N, 2C) via mean and max over valid positions only."""
    mask = valid.unsqueeze(1).float()                    # (N, 1, L)
    denom = mask.sum(dim=2).clamp_min(1e-8)
    mean = (h * mask).sum(dim=2) / denom
    maximum = h.masked_fill(mask == 0, float("-inf")).max(dim=2).values
    maximum = torch.nan_to_num(maximum, neginf=0.0)
    return torch.cat([mean, maximum], dim=1)


# Head layout a checkpoint was trained with; checked on load.
HEAD_SET = "ACP-pair-dg4-pred-noseq"


_WC_STRONG = [("G", "C"), ("C", "G")]


_WC_WEAK = [("A", "T"), ("T", "A")]


def wc_matrices(alphabet: int) -> torch.Tensor:
    """(2, A, A) Watson-Crick indicator: channel 0 strong (G-C), 1 weak (A-T)."""
    m = torch.zeros(2, alphabet, alphabet)
    for ch, pairs in enumerate((_WC_STRONG, _WC_WEAK)):
        for x, y in pairs:
            i, j = BASES.index(x), BASES.index(y)
            if max(i, j) < alphabet:
                m[ch, i, j] = 1.0
    return m


@torch.no_grad()
def head_a_score_dg(model: nn.Module, loader: DataLoader, device: torch.device,
                    bin_edges: list[np.ndarray], limits: list[float],
                    widths: list[float]) -> dict[str, np.ndarray]:
    """Head A: probability that a structure is at least as stable as its limit."""
    model.eval()
    masks = [torch.tensor(below_class_mask(e, lim), device=device)
             for e, lim in zip(bin_edges, limits)]
    centers = [torch.tensor(dg_bin_centers(e, w), device=device, dtype=torch.float32)
               for e, w in zip(bin_edges, widths)]

    breach_parts, dg_parts = [], []
    for (seq,) in loader:
        seq = seq.to(device).float()
        out = model(seq, apply_mask=False)
        batch_breach, batch_dg = [], []
        for j, logits in enumerate(out["dg_bin_logits"]):
            p = torch.softmax(logits, dim=1)
            batch_breach.append(p[:, masks[j]].sum(dim=1))
            batch_dg.append((p * centers[j]).sum(dim=1))
        breach_parts.append(torch.stack(batch_breach, dim=1).cpu().numpy())
        dg_parts.append(torch.stack(batch_dg, dim=1).cpu().numpy())

    p_breach = np.concatenate(breach_parts, axis=0)          # (N, S)
    rule = p_breach[:, list(HEAD_A_RULE_INDICES)]
    structural = 1.0 - np.prod(1.0 - rule, axis=1)
    return {"score": structural, "structural": structural, "p_breach": p_breach,
            "dg_pred": np.concatenate(dg_parts, axis=0)}


# Head C's structural coordinates are Head A's predicted dG, re-binned on Head C's grid.
def dg_center_tensors(bin_edges: list[np.ndarray], widths: list[float],
                      device: torch.device) -> list[torch.Tensor]:
    """Head A's per-class representative dG, one tensor per structure."""
    return [torch.tensor(dg_bin_centers(e, w), device=device, dtype=torch.float32)
            for e, w in zip(bin_edges, widths)]


def dg_edge_tensors(binners: list[Binner], device: torch.device) -> list[torch.Tensor]:
    """Head C's bin edges for its structural coordinates, on `device`."""
    return [torch.tensor(binners[j].edges, device=device, dtype=torch.float32)
            for j in range(SEQ_PROPERTY_COUNT, len(binners))]


def expected_dg(dg_bin_logits: list[torch.Tensor],
                centers: list[torch.Tensor]) -> torch.Tensor:
    """(N, S) mean free energy under Head A's own bin posterior."""
    return torch.stack([(torch.softmax(logits, dim=1) * c).sum(dim=1)
                        for logits, c in zip(dg_bin_logits, centers)], dim=1)


def predicted_dg_classes(seq_cls: torch.Tensor, dg_bin_logits: list[torch.Tensor],
                         centers: list[torch.Tensor],
                         edges: list[torch.Tensor]) -> torch.Tensor:
    """(N, 6) sequence classes + Head A's predicted dG classes -> (N, 8)."""
    dg_pred = expected_dg([dg_bin_logits[j].detach() for j in HEAD_C_DG_INDICES],
                          [centers[j] for j in HEAD_C_DG_INDICES])
    # .contiguous(): a column slice of (N, S) is strided, and torch.bucketize
    # copies it internally rather than reading it in place.
    dg_cls = torch.stack([torch.bucketize(dg_pred[:, j].contiguous(), e, right=True)
                          for j, e in enumerate(edges)], dim=1)
    return torch.cat([seq_cls, dg_cls], dim=1)


@torch.no_grad()
def property_class_matrix(model: nn.Module, seq: torch.Tensor, seq_cls: torch.Tensor,
                          device: torch.device, centers: list[torch.Tensor],
                          edges: list[torch.Tensor], batch_size: int = 256) -> torch.Tensor:
    """(N, 8) Head C input for a whole split, assembled once."""
    model.eval()
    out = []
    for i in range(0, len(seq), batch_size):
        batch = seq[i:i + batch_size].to(device).float()
        logits = model(batch, apply_mask=False)["dg_bin_logits"]
        out.append(predicted_dg_classes(
            seq_cls[i:i + batch_size].to(device), logits, centers, edges).cpu())
    return torch.cat(out, dim=0)


def _groups(channels: int) -> int:
    return next((g for g in (8, 4, 2, 1) if channels % g == 0), 1)


class PairwiseStem(nn.Module):
    """(L, L) pair features from complementarity, per-base features and distance."""

    def __init__(self, channels: int, alphabet: int, seq_len: int,
                 pair_channels: int = 32, n_layers: int = 3, max_dist: int = 64):
        super().__init__()
        self.pair_channels = pair_channels
        self.seq_len = seq_len
        self.out_dim = 2 * pair_channels

        self.register_buffer("wc", wc_matrices(alphabet), persistent=False)

        self.sum_proj = nn.Conv1d(channels, pair_channels, kernel_size=1)
        self.mul_proj = nn.Conv1d(channels, pair_channels, kernel_size=1)
        self.dist_emb = nn.Embedding(max_dist + 1, pair_channels)
        nn.init.normal_(self.dist_emb.weight, std=0.02)

        self.stem_in = nn.Conv2d(2 + 3 * pair_channels, pair_channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(pair_channels, pair_channels, kernel_size=3, padding=1),
                nn.GroupNorm(_groups(pair_channels), pair_channels),
                nn.GELU(),
                nn.Conv2d(pair_channels, pair_channels, kernel_size=3, padding=1),
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.GroupNorm(_groups(pair_channels), pair_channels)
        self.pair_logit = nn.Conv2d(pair_channels, 1, kernel_size=1)
        self.unpaired_logit = nn.Conv1d(self.out_dim, 1, kernel_size=1)

        idx = torch.arange(seq_len)
        dist = (idx[:, None] - idx[None, :]).abs()
        self.register_buffer("dist_idx", dist.clamp(max=max_dist), persistent=False)
        self.register_buffer("span_ok", dist >= MIN_HAIRPIN_SPAN, persistent=False)

    def forward(self, seq: torch.Tensor, h: torch.Tensor,
                valid: torch.Tensor) -> dict[str, torch.Tensor]:
        n = valid.shape[0]
        wc = torch.einsum("nia,cab,njb->ncij", seq, self.wc.to(seq.dtype), seq)

        a = self.sum_proj(h)
        b = self.mul_proj(h)
        x = self.stem_in(torch.cat([
            wc,
            a.unsqueeze(3) + a.unsqueeze(2),
            b.unsqueeze(3) * b.unsqueeze(2),
            self.dist_emb(self.dist_idx).permute(2, 0, 1).unsqueeze(0).expand(n, -1, -1, -1),
        ], dim=1))
        for block in self.blocks:
            x = F.gelu(x + block(x))
        x = self.norm(x)

        pair_ok = valid.unsqueeze(2) & valid.unsqueeze(1) & self.span_ok.unsqueeze(0)
        x = x * pair_ok.unsqueeze(1).to(x.dtype)

        logits = self.pair_logit(x).squeeze(1)
        logits = 0.5 * (logits + logits.transpose(1, 2))
        logits = logits.masked_fill(~pair_ok, torch.finfo(logits.dtype).min)

        masked = x.masked_fill(~pair_ok.unsqueeze(1), torch.finfo(x.dtype).min)
        pos_max = masked.max(dim=3).values
        pos_max = torch.where(pair_ok.any(dim=2).unsqueeze(1), pos_max, torch.zeros_like(pos_max))
        pos_mean = x.sum(dim=3) / pair_ok.sum(dim=2).clamp_min(1).unsqueeze(1).to(x.dtype)
        pos_feats = torch.cat([pos_max, pos_mean], dim=1)

        logits = torch.cat([logits, self.unpaired_logit(pos_feats).transpose(1, 2)], dim=2)
        return {"pair_map_logits": logits, "pos_feats": pos_feats,
                "summary": masked_pool(pos_feats, valid)}


# Head P: S_P is the largest per-position pairing probability (max_paired).
PAIR_FEATURE_NAMES = [
    # -- marginal: collapse (L, L+1) to a per-position "is this base paired"
    "mean_paired", "max_paired", "sum_paired", "entropy", "n_paired", "max_run",
    # -- structural: read the matrix as a fold rather than as a per-position rate
    "max_pair_prob", "helix_len", "helix_mass", "concentration", "min_loop", "mean_span",
    # -- positional: where the pairing sits.
    "p3_1", "p3_3", "p3_5", "stem3_max", "p5_3", "asym_3v5",
]


PAIR_SCORE_FEATURE = "max_paired"


def _pair_feature_row(sub: np.ndarray) -> list[float]:
    """One primer's features from its (V, V+1) pair distribution over real positions."""
    v_len = sub.shape[0]
    if v_len < 2:
        return [0.0] * len(PAIR_FEATURE_NAMES)

    unpaired = sub[:, -1]
    paired = 1.0 - unpaired
    pair_mat = sub[:, :-1]

    ent = -(paired * np.log(paired + 1e-9) + unpaired * np.log(unpaired + 1e-9))
    best_p = pair_mat.max(axis=1)
    # Paired = one partner carries more than half the mass (as in hard_partner).
    partner = np.where(best_p > 0.5, pair_mat.argmax(axis=1), -1)

    # Longest contiguous helix: i pairs j and i+1 pairs j-1 is a stacked pair.
    helix = run = 0
    for i in range(v_len - 1):
        if partner[i] >= 0 and partner[i + 1] == partner[i] - 1:
            run += 1
            helix = max(helix, run)
        else:
            run = 0
    helix = helix + 1 if helix else 0

    longest = run = 0
    for is_paired in paired > 0.5:
        run = run + 1 if is_paired else 0
        longest = max(longest, run)

    # Span features are NaN, not 0, when no position has a confident partner.
    has_partner = partner >= 0
    spans = (np.abs(partner[has_partner] - np.flatnonzero(has_partner))
             if has_partner.any() else np.array([np.nan]))
    k3 = min(5, v_len)

    return [
        float(paired.mean()), float(paired.max()), float(paired.sum()),
        float(ent.mean()), float(has_partner.sum()), float(longest),
        float(pair_mat.max()), float(helix),
        float(best_p[has_partner].sum()) if has_partner.any() else 0.0,
        float(best_p.sum() / max(paired.sum(), 1e-9)),
        float(np.min(spans)), float(np.mean(spans)),   # NaN when nothing pairs
        float(paired[-1]), float(paired[-3:].mean()), float(paired[-k3:].mean()),
        float(pair_mat[-k3:, :].max()), float(paired[:3].mean()),
        float(paired[-k3:].mean() - paired[:k3].mean()),
    ]


@torch.no_grad()
def pair_map_features(model: nn.Module, loader: DataLoader,
                      device: torch.device) -> dict[str, object]:
    """(N, len(PAIR_FEATURE_NAMES)) summary of the predicted pair map."""
    model.eval()
    rows = []
    for (seq,) in loader:
        seq = seq.to(device).float()
        out = model(seq, apply_mask=False)
        pair_map = torch.softmax(out["pair_map_logits"], dim=-1).cpu().numpy()
        valid = out["valid"].cpu().numpy()
        seq_len = pair_map.shape[1]
        for n in range(len(pair_map)):
            keep = np.flatnonzero(valid[n])
            sub = pair_map[n][np.ix_(keep, list(keep) + [seq_len])]
            rows.append(_pair_feature_row(sub))
    return {"features": np.asarray(rows, dtype=np.float64),
            "names": list(PAIR_FEATURE_NAMES)}


@torch.no_grad()
def head_p_score(model: nn.Module, loader: DataLoader,
                 device: torch.device) -> dict[str, np.ndarray]:
    """Head P: the pair map's own decision score, plus the full feature block."""
    out = pair_map_features(model, loader, device)
    features = out["features"]
    return {"score": features[:, PAIR_FEATURE_NAMES.index(PAIR_SCORE_FEATURE)],
            "features": features, "names": out["names"]}


# Model
class PrimerEncoder(nn.Module):
    """Shared encoder + pairwise stem with the pairing, free-energy (A) and density (C)
    heads.
    """

    def __init__(self, seq_len: int, alphabet: int, bin_counts: list[int],
                 property_class_counts: list[int], channels: int = 96,
                 pair_channels: int = 32, pair_layers: int = 3,
                 density_dim: int = 96, density_layers: int = 2,
                 density_heads: int = 4):
        super().__init__()
        self.seq_len = seq_len
        self.alphabet = alphabet
        self.channels = channels
        self.bin_counts = list(bin_counts)

        self.encoder = SequenceEncoder(alphabet, channels)
        self.pairwise = PairwiseStem(channels, alphabet, seq_len,
                                     pair_channels=pair_channels, n_layers=pair_layers)
        pair_dim = self.pairwise.out_dim

        # ---- per-position 3-class pairing head and Head A's free-energy bins
        self.pair_head = nn.Conv1d(channels + pair_dim, 3, kernel_size=1)
        self.dg_bin_heads = nn.ModuleList(
            [nn.Linear(2 * channels + 2 * pair_dim, k) for k in self.bin_counts])

        # ---- Head C: a model over the property classes alone
        self.density = PropertyDensity(
            property_class_counts,
            d_model=density_dim,
            n_layers=density_layers,
            n_heads=density_heads,
        )

    def forward(self, seq: torch.Tensor, apply_mask: bool = False) -> dict[str, torch.Tensor]:
        valid = seq.sum(dim=-1) > 0
        h = self.encoder(seq)
        pw = self.pairwise(seq, h, valid)
        z = torch.cat([h, pw["pos_feats"]], dim=1)
        pooled = torch.cat([masked_pool(h, valid), pw["summary"]], dim=1)

        return {
            "pair_logits": self.pair_head(z).transpose(1, 2),
            "pair_map_logits": pw["pair_map_logits"],
            "dg_bin_logits": [head(pooled) for head in self.dg_bin_heads],
            "h": z,
            "valid": valid,
            "pooled": pooled,
        }


def build_model_from_checkpoint(ckpt: dict) -> PrimerEncoder:
    """Build an untrained PrimerEncoder with the layout stored in `ckpt`."""
    if ckpt.get("head_set") != HEAD_SET:
        raise ValueError(
            f"checkpoint head_set={ckpt.get('head_set')!r} is not {HEAD_SET!r}: it was "
            "trained with a different head layout and cannot be scored by this code. "
            "Retrain with:  python -m olivia_encoder.train"
        )
    return PrimerEncoder(
        seq_len=ckpt["seq_len"], alphabet=ckpt["alphabet"], bin_counts=ckpt["bin_counts"],
        property_class_counts=ckpt["property_class_counts"], channels=ckpt["channels"],
        pair_channels=ckpt["pair_channels"], pair_layers=ckpt["pair_layers"],
        density_dim=ckpt["density_dim"], density_layers=ckpt["density_layers"],
        density_heads=ckpt["density_heads"],
    )


def scalers_from_checkpoint(ckpt: dict) -> tuple[TermScaler, TermScaler]:
    """The two frozen TermScalers, or a clear error if the checkpoint lacks them."""
    try:
        return (TermScaler.from_dict(ckpt["head_c_marg_scaler"]),
                TermScaler.from_dict(ckpt["head_c_inter_scaler"]))
    except KeyError as exc:
        raise KeyError(
            f"checkpoint has no {exc.args[0]!r}. Head C's scores are per-coordinate "
            "standardised maxima and the standardisation is fitted on the training "
            "split, so the moments are part of the model in the same way the bin "
            "edges are. Retrain."
        ) from exc
