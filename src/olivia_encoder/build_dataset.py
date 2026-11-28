"""Step 2 of dataset preparation: split a primer CSV 70/15/15 and encode it as tensors.

    python -m olivia_encoder.build_dataset --input primers_calculated.csv
    --output-dir data/embedded/true_data_all_t37
    python -m olivia_encoder.build_dataset --input eval_calculated.csv --output-dir
    data/embedded/eval_set \\
        --reference-dir data/embedded/true_data_all_t37

--reference-dir embeds an evaluation corpus against an existing dataset's schema.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from .sequence import clean_sequence, sequence_to_tensor

SEED = 42
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+")


def load_reference_schema(reference_dir: Path) -> dict[str, Any]:
    with (reference_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return {"sequence_length": int(metadata.get("sequence_length", 1))}


# Utility Functions

def parse_numeric(value: Any) -> float:
    """Parses numeric values, handling strings, NaNs, and averaging ranges (e.g., '40-45')."""
    if pd.isna(value):
        return math.nan
    cleaned = str(value).replace(",", "")
    matches = _FLOAT_RE.findall(cleaned)
    if not matches:
        return math.nan
    return sum(float(m) for m in matches) / len(matches)


# Data Splitting

def normalize_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "cleaned_primer" not in normalized.columns and "primer" in normalized.columns:
        normalized["cleaned_primer"] = normalized["primer"]
    if "primer_length" not in normalized.columns:
        source_sequences = normalized.get(
            "cleaned_primer", normalized.get("primer", pd.Series(index=normalized.index, dtype=object))
        )
        normalized["primer_length"] = source_sequences.fillna("").map(
            lambda value: len(clean_sequence(value)) if str(value).strip() else math.nan
        )
    if "annealing_temp" not in normalized.columns:
        normalized["annealing_temp"] = math.nan
    return normalized


def split_frame(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the dataframe into Train (70%), Validation (15%), and Test (15%) sets."""
    train_frame, temp_frame = train_test_split(frame, test_size=0.30, random_state=seed)
    val_frame, test_frame = train_test_split(temp_frame, test_size=0.50, random_state=seed)
    return train_frame.reset_index(drop=True), val_frame.reset_index(drop=True), test_frame.reset_index(drop=True)


# Filtering

def _as_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    return series.apply(parse_numeric)


def filter_invalid_primers(frame: pd.DataFrame) -> pd.DataFrame:
    """Drops primers with out-of-range sequence length or annealing temperature."""
    filtered = frame.copy()
    if "primer_length" in filtered.columns:
        primer_lengths = _as_numeric(filtered["primer_length"])
    else:
        source_sequences = filtered.get("cleaned_primer", filtered.get("primer", pd.Series(index=filtered.index, dtype=object)))
        primer_lengths = source_sequences.fillna("").map(lambda value: len(clean_sequence(value)) if str(value).strip() else math.nan)

    if "annealing_temp" in filtered.columns:
        annealing_temps = _as_numeric(filtered["annealing_temp"])
    else:
        annealing_temps = pd.Series(math.nan, index=filtered.index)

    valid_mask = pd.Series(True, index=filtered.index)
    valid_mask &= (primer_lengths.isna() | primer_lengths.between(10, 40, inclusive="both"))
    valid_mask &= (annealing_temps.isna() | annealing_temps.between(45, 75, inclusive="both"))
    return filtered.loc[valid_mask].reset_index(drop=True)


# Sequence Encoding

def build_sequence_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    reference_max_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Encodes sequences into 3D tensors with ALL-ZERO padding.
    """
    train_clean = [clean_sequence(s) for s in train_df["cleaned_primer"]]
    val_clean = [clean_sequence(s) for s in val_df["cleaned_primer"]]
    test_clean = [clean_sequence(s) for s in test_df["cleaned_primer"]]

    max_length = reference_max_length if reference_max_length is not None else max((len(s) for s in train_clean), default=1)
    max_length = max(max_length, 1)

    def encode(clean_list: list[str]) -> np.ndarray:
        return np.stack([sequence_to_tensor(s, max_length) for s in clean_list])

    return encode(train_clean), encode(val_clean), encode(test_clean), int(max_length)


# Orchestration & Storage

def save_split_bundle(output_dir: Path, split_name: str, frame: pd.DataFrame, sequence: np.ndarray) -> None:
    """Packages the sequence tensor and source rows into a PyTorch dictionary and saves to disk."""
    bundle = {
        "primer_sequence": torch.tensor(sequence, dtype=torch.float32),
        "frame": frame,
    }
    torch.save(bundle, output_dir / f"{split_name}.pt")
    frame.to_csv(output_dir / f"{split_name}.csv", index=False)


def preprocess_data(
    input_path: Path,
    output_dir: Path,
    seed: int = SEED,
    reference_dir: Path | None = None,
) -> dict[str, Any]:
    """Load, filter, split, encode and save one dataset; returns its metadata."""
    frame = normalize_input_frame(pd.read_csv(input_path))
    reference_schema = load_reference_schema(reference_dir) if reference_dir is not None else None

    frame = filter_invalid_primers(frame)
    train_df, val_df, test_df = split_frame(frame, seed=seed)

    train_seq, val_seq, test_seq, max_seq_len = build_sequence_matrices(
        train_df,
        val_df,
        test_df,
        reference_max_length=(reference_schema["sequence_length"] if reference_schema else None),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_split_bundle(output_dir, "train", train_df, train_seq)
    save_split_bundle(output_dir, "val", val_df, val_seq)
    save_split_bundle(output_dir, "test", test_df, test_seq)

    metadata = {
        "seed": seed,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "sequence_length": max_seq_len,
        "sequence_encoding": "multihot",
        "sequence_feature_dim": 4,
        "padding": "all_zero",
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a primer CSV and encode it as tensors.")
    parser.add_argument("--input", type=Path, required=True,
                        help="CSV written by prepare_primers.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for train/val/test .pt, .csv and metadata.json.")
    parser.add_argument("--reference-dir", type=Path, default=None,
                        help="Embed against this existing dataset's schema (for evaluation corpora).")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for the split.")
    args = parser.parse_args()
    metadata = preprocess_data(args.input, args.output_dir, seed=args.seed,
                               reference_dir=args.reference_dir)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
