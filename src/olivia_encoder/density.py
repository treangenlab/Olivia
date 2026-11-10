"""Head C (paper: M_C): a joint density over the eight primer properties.

A small transformer over property tokens, trained order-agnostically, gives
every conditional p(x_j | x_S); an independent marginal per property is trained
alongside. Two scores, each the max over properties of a term standardised on
the training split:

    C_marg  = max_j z(-log p(x_j))                       rare property value
    C_inter = max_j z(-log p(x_j | x_-j) + log p(x_j))   rare given the others
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import SEED
from .properties import N_PROPERTIES


class PropertyDensity(nn.Module):
    """p(properties), any conditional, one model."""

    def __init__(self, class_counts: list[int], d_model: int = 96, n_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.class_counts = list(class_counts)
        self.n_props = len(class_counts)
        self.d_model = d_model

        self.value_emb = nn.ModuleList([nn.Embedding(k, d_model) for k in class_counts])
        self.unobserved = nn.Parameter(torch.zeros(self.n_props, d_model))
        self.prop_id = nn.Parameter(torch.zeros(self.n_props, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.ModuleList([nn.Linear(d_model, k) for k in class_counts])

        self.marg_logits = nn.ParameterList([nn.Parameter(torch.zeros(k)) for k in class_counts])

        nn.init.normal_(self.unobserved, std=0.02)
        nn.init.normal_(self.prop_id, std=0.02)

    def forward(self, cls: torch.Tensor, observed: torch.Tensor) -> list[torch.Tensor]:
        """cls (N, D) long, observed (N, D) bool -> D logit tensors (N, K_j)."""
        toks = []
        for j in range(self.n_props):
            e = self.value_emb[j](cls[:, j])
            e = torch.where(observed[:, j].unsqueeze(1), e, self.unobserved[j].expand_as(e))
            toks.append(e + self.prop_id[j])
        x = torch.stack(toks, dim=1)                                  # (N, D, d)
        pad = torch.zeros(cls.shape[0], self.n_props, dtype=torch.bool, device=cls.device)
        h = self.backbone(x, src_key_padding_mask=pad)
        return [self.out[j](h[:, j]) for j in range(self.n_props)]

    def marginal_logprob(self, cls: torch.Tensor) -> torch.Tensor:
        """(N,) sum of independent per-property log-probabilities."""
        total = torch.zeros(cls.shape[0], device=cls.device)
        for j in range(self.n_props):
            lp = F.log_softmax(self.marg_logits[j], dim=0)
            total = total + lp[cls[:, j]]
        return total


def joint_density_loss(density: PropertyDensity,
                       cls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Order-agnostic NLL of p(properties), and the independent-marginal NLL."""
    n, d = cls.shape
    k = torch.randint(0, d, (n,), device=cls.device)
    rank = torch.rand(n, d, device=cls.device).argsort(dim=1).argsort(dim=1)
    observed = rank < k.unsqueeze(1)
    hidden = ~observed

    ce = torch.stack(
        [F.cross_entropy(logits, cls[:, j], reduction="none")
         for j, logits in enumerate(density(cls, observed))], dim=1)
    hidden_f = hidden.float()
    # D * mean over hidden == (D / (D-k)) * sum over hidden. k <= D-1 guarantees
    # at least one hidden property, so this never divides by zero.
    per_sample = float(d) * (ce * hidden_f).sum(dim=1) / hidden_f.sum(dim=1).clamp_min(1.0)

    marg = torch.stack(
        [F.cross_entropy(density.marg_logits[j].unsqueeze(0).expand(n, -1),
                         cls[:, j], reduction="none") for j in range(d)], dim=1).sum(dim=1)

    return per_sample.mean(), marg.mean()


def chain_rule_logprob(density: PropertyDensity, cls: torch.Tensor,
                       order: list[int]) -> torch.Tensor:
    """Exact log p under one autoregressive order (D forward passes)."""
    n, d = cls.shape
    observed = torch.zeros(n, d, dtype=torch.bool, device=cls.device)
    total = torch.zeros(n, device=cls.device)
    for j in order:
        logits = density(cls, observed)
        lp = F.log_softmax(logits[j], dim=1)
        total = total + lp.gather(1, cls[:, j].unsqueeze(1)).squeeze(1)
        observed[:, j] = True
    return total


@torch.no_grad()
def all_masked_logits(density: PropertyDensity, device: torch.device) -> list[torch.Tensor]:
    """The density's logits with nothing observed: the model's own p(x_j)."""
    cls = torch.zeros(1, N_PROPERTIES, dtype=torch.long, device=device)
    observed = torch.zeros(1, N_PROPERTIES, dtype=torch.bool, device=device)
    return density(cls, observed)


@torch.no_grad()
def head_c_terms(density: PropertyDensity, cls: torch.Tensor, device: torch.device,
                 batch_size: int = 1024) -> dict[str, np.ndarray]:
    """Per-coordinate surprisal terms for a whole split."""
    d = cls.shape[1]
    marg_rows, cond_rows = [], []
    for (batch,) in DataLoader(TensorDataset(cls), batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        n = len(batch)
        marg_rows.append(torch.stack(
            [-F.log_softmax(density.marg_logits[j], dim=0)[batch[:, j]] for j in range(d)],
            dim=1).cpu())
        cond = []
        for j in range(d):
            observed = torch.ones(n, d, dtype=torch.bool, device=device)
            observed[:, j] = False              # everything except j
            logits = density(batch, observed)[j]
            cond.append(-F.log_softmax(logits, dim=1).gather(1, batch[:, j:j + 1]).squeeze(1))
        cond_rows.append(torch.stack(cond, dim=1).cpu())

    s_marg = torch.cat(marg_rows).numpy()
    s_cond = torch.cat(cond_rows).numpy()
    return {"s_marg": s_marg, "s_cond": s_cond, "d": s_cond - s_marg}


class TermScaler:
    """Per-coordinate mean and sd of a surprisal term over known-good primers."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float64)
        # A coordinate can be constant on the training split -- gc_clamp on a narrow
        # corpus, say.
        self.std = np.maximum(np.asarray(std, dtype=np.float64), 1e-6)

    @classmethod
    def fit(cls, terms: np.ndarray) -> "TermScaler":
        return cls(terms.mean(axis=0), terms.std(axis=0))

    def z(self, terms: np.ndarray) -> np.ndarray:
        return (np.asarray(terms, dtype=np.float64) - self.mean) / self.std

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "TermScaler":
        return cls(np.asarray(d["mean"]), np.asarray(d["std"]))


def head_c_scores(terms: dict[str, np.ndarray], marg_scaler: TermScaler,
                  inter_scaler: TermScaler, n_orders: int = 4,
                  density: PropertyDensity | None = None,
                  cls: torch.Tensor | None = None,
                  device: torch.device | None = None) -> dict[str, np.ndarray]:
    """Head C's two decision scores, plus the diagnostics they decompose."""
    z_marg = marg_scaler.z(terms["s_marg"])
    z_inter = inter_scaler.z(terms["d"])

    out = {
        "c_marg": z_marg.max(axis=1),
        "c_inter": z_inter.max(axis=1),
        "c_marg_which": z_marg.argmax(axis=1),
        "c_inter_which": z_inter.argmax(axis=1),
        "z_marg": z_marg,
        "z_inter": z_inter,
        "surprisal_marg": terms["s_marg"].sum(axis=1),
    }

    if density is not None:
        rng = np.random.default_rng(SEED)
        orders = [list(rng.permutation(cls.shape[1])) for _ in range(n_orders)]
        joint = []
        with torch.no_grad():
            for (batch,) in DataLoader(TensorDataset(cls), batch_size=1024, shuffle=False):
                batch = batch.to(device)
                joint.append((-torch.stack(
                    [chain_rule_logprob(density, batch, o) for o in orders]
                ).mean(0)).cpu().numpy())
        out["surprisal_joint"] = np.concatenate(joint)
        out["synergy"] = out["surprisal_joint"] - out["surprisal_marg"]
    return out


@torch.no_grad()
def per_property_interaction_gain(terms: dict[str, np.ndarray],
                                  property_names: list[str]) -> pd.DataFrame:
    """Per property: marginal vs conditional surprisal; gain_nats = MI(x_j ; x_{-j})."""
    marg = terms["s_marg"].mean(axis=0)
    cond = terms["s_cond"].mean(axis=0)
    return pd.DataFrame({
        "property": property_names,
        "marginal_nats": marg,
        "given_others_nats": cond,
        "gain_nats": marg - cond,
        "gain_frac": (marg - cond) / np.maximum(marg, 1e-8),
    })
