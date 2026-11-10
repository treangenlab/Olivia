"""Embedded datasets and the NUPACK target cache.

A dataset directory (written by build_dataset.py) holds train/val/test .pt
bundles plus metadata.json; its nupack/nupack_cache.npz (written by
nupack_targets.py) holds the free energies and pairing maps the encoder is
trained against.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .sequence import (
    clean_sequence,
)

SEED = 42


class SplitBundle:
    """One split: one-hot sequences, tabular features, labels and the source frame."""

    def __init__(self, path: Path):
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(path, map_location="cpu")
        self.tabular = data["tabular"].float()
        self.sequence = data.get("primer_sequence", data.get("sequence")).float()
        self.labels = data["labels"].long()
        self.frame = data["frame"]


def load_metadata(embedded_dir: Path) -> dict[str, object]:
    with (embedded_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_bundles(embedded_dir: Path) -> dict[str, SplitBundle]:
    return {split: SplitBundle(embedded_dir / f"{split}.pt") for split in ("train", "val", "test")}


def bundle_sequences(bundle: SplitBundle) -> list[str]:
    col = "cleaned_primer" if "cleaned_primer" in bundle.frame.columns else "primer"
    return [clean_sequence(s) for s in bundle.frame[col]]


UNPAIRED_PARTNER = -1


# Every free energy `nupack_targets.py` caches, in cache-column order.
DG_CACHE_FIELDS = ("hairpin_dg", "hairpin_dg_mfe", "homodimer_dg", "homodimer_dg_mfe")


class NupackCache:
    """Free energies and MFE pairing maps precomputed by `nupack_targets.py`."""

    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(
                f"NUPACK cache {path} not found. Build it with:\n"
                f"  python -m olivia_encoder.nupack_dump_sequences --embedded-dir <dir>\n"
                f"  python -m olivia_encoder.nupack_targets        --embedded-dir <dir>"
            )
        data = np.load(path, allow_pickle=True)
        self.path = path
        self.energies = data["energies"].astype(np.float64)
        self.partner = data["partner"].astype(np.int64)
        self.fields = [str(f) for f in data["fields"]]
        self.max_len = int(data["max_len"])
        self.fold_temp_c = float(data["fold_temp_c"])
        self.sodium_m = float(data["sodium_m"])
        self.magnesium_m = float(data["magnesium_m"])
        self.nupack_version = str(data["nupack_version"])
        self.index = {str(s): i for i, s in enumerate(data["sequences"])}

        # The pairing head trains on the ensemble pair matrix P(i, j), not the
        # MFE pair list, so a cache without it is an error rather than a fallback.
        if "pair_top_idx" not in data.files:
            raise KeyError(
                f"{path.name} has no ensemble pair matrix ('pair_top_idx'). Re-fold it with:\n"
                f"  python -m olivia_encoder.nupack_dump_sequences --embedded-dir <dir>\n"
                f"  python -m olivia_encoder.nupack_targets        --embedded-dir <dir>"
            )
        self.pair_top_idx = data["pair_top_idx"].astype(np.int64)
        self.pair_top_prob = data["pair_top_prob"].astype(np.float32)
        self.pair_unpaired = data["pair_unpaired"].astype(np.float32)
        self.pair_top_k = int(data["pair_top_k"])
        self.pair_prob_min = float(data["pair_prob_min"])

    def rows(self, seqs: list[str]) -> np.ndarray:
        try:
            return np.array([self.index[s] for s in seqs], dtype=np.int64)
        except KeyError as exc:
            missing = [s for s in seqs if s not in self.index]
            raise KeyError(
                f"{len(missing)} sequence(s) are not in {self.path.name}, e.g. {missing[:3]}. "
                f"The cache is stale -- re-run nupack_dump_sequences then nupack_targets."
            ) from exc

    def energies_for(self, seqs: list[str], fields: tuple[str, ...]) -> np.ndarray:
        cols = [self.fields.index(f) for f in fields]
        return self.energies[np.ix_(self.rows(seqs), cols)]

    def partners_for(self, seqs: list[str], seq_len: int) -> np.ndarray:
        """(N, seq_len) MFE partner index, -1 unpaired, truncated to seq_len."""
        raw = self.partner[self.rows(seqs)]
        out = np.full((len(seqs), seq_len), UNPAIRED_PARTNER, dtype=np.int64)
        keep = min(seq_len, raw.shape[1])
        out[:, :keep] = raw[:, :keep]
        # A partner past the truncation point cannot be represented, so drop the
        # half-pair rather than leave an index pointing into padding.
        out[out >= seq_len] = UNPAIRED_PARTNER
        return out

    def pair_probs_for(self, seqs: list[str], seq_len: int
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ensemble pair probabilities, truncated to seq_len."""
        rows = self.rows(seqs)
        n, k = len(seqs), self.pair_top_k
        keep = min(seq_len, self.max_len)

        idx = np.full((n, seq_len, k), -1, dtype=np.int64)
        prob = np.zeros((n, seq_len, k), dtype=np.float32)
        # Padding positions are not valid anywhere downstream, but "certainly
        # unpaired" is the right filler for a position that does not exist.
        unpaired = np.ones((n, seq_len), dtype=np.float32)

        idx[:, :keep] = self.pair_top_idx[rows][:, :keep]
        prob[:, :keep] = self.pair_top_prob[rows][:, :keep]
        unpaired[:, :keep] = self.pair_unpaired[rows][:, :keep]

        # Same truncation rule as partners_for: a partner past seq_len cannot be
        # represented, so the slot is emptied rather than left dangling.
        gone = idx >= seq_len
        idx[gone] = -1
        prob[gone] = 0.0
        return idx, prob, unpaired
