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
from sklearn.preprocessing import StandardScaler

from .sequence import clean_sequence, sequence_to_tensor

SEED = 42
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+")

TABULAR_NUMERIC_COLUMNS = [
    "annealing_temp", "Mg_concentration", "primer_concentration",
    "cycle_num", "amplicon_size", "calculated_annealing_temp",
    "primer_length", "gc_content", "delta_g_kcal", "source_count",
    "hairpin_found", "hairpin_dg", "hairpin_tm",
    "homodimer_found", "homodimer_dg", "homodimer_tm",
]

# 'No fold found' columns: kept as the sentinel, not median-imputed.
STRUCTURAL_SENTINEL_COLUMNS = ["hairpin_tm", "homodimer_tm"]
STRUCTURAL_SENTINEL_VALUE = -99.0

TABULAR_CATEGORICAL_COLUMNS = ["primer_type", "template_type"]


def load_reference_schema(reference_dir: Path) -> dict[str, Any]:
    reference_csv = reference_dir / "train.csv"
    reference_meta = reference_dir / "metadata.json"
    reference_frame = pd.read_csv(reference_csv)
    with reference_meta.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    reference_cat_cols = list(
        pd.get_dummies(reference_frame[TABULAR_CATEGORICAL_COLUMNS].fillna("missing").astype(str)).columns
    )
    return {
        "cat_cols": reference_cat_cols,
        "max_tax_len": int(metadata.get("max_taxonomy_padding_length", 1)),
        "sequence_length": int(metadata.get("sequence_length", 1)),
    }


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


def parse_taxonomy_list(value: Any) -> list[str]:
    """Parses stringified lists like '[10265.0, 186805.0]' into clean string IDs ['10265', '186805']."""
    if pd.isna(value) or str(value).strip() in ("", "[]", "nan"):
        return []
    cleaned = re.sub(r"[\[\]\s]", "", str(value))
    cleaned = re.sub(r"\.0", "", cleaned)
    return [x for x in cleaned.split(",") if x]


# Taxonomy Tokenizer

def build_taxonomy_vocab(train_series: pd.Series) -> dict[str, int]:
    """Creates a vocabulary mapping from unique taxonomy strings to integer tokens (train only)."""
    unique_ids = set()
    for row in train_series:
        unique_ids.update(parse_taxonomy_list(row))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for idx, tax_id in enumerate(sorted(unique_ids), start=2):
        vocab[tax_id] = idx
    print(f"Built taxonomy vocabulary with {len(vocab)} tokens (including <PAD> and <UNK>).")
    return vocab


def tokenize_and_pad_taxonomy(series: pd.Series, vocab: dict[str, int], max_len: int) -> np.ndarray:
    """Transforms taxonomic string lists into a fixed-width padded token matrix."""
    matrix = np.zeros((len(series), max_len), dtype=np.int64)
    for row_idx, val in enumerate(series):
        tokens = parse_taxonomy_list(val)[:max_len]
        for col_idx, token in enumerate(tokens):
            matrix[row_idx, col_idx] = vocab.get(token, vocab["<UNK>"])
    return matrix


# Data Splitting

def prepare_frame(df: pd.DataFrame, label: int) -> pd.DataFrame:
    frame = df.copy().reset_index(drop=True)
    frame["label"] = label
    return frame


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


def split_labeled_frame(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the dataframe into Train (70%), Validation (15%), and Test (15%) sets."""
    stratify = frame["label"] if frame["label"].nunique() > 1 else None
    train_frame, temp_frame = train_test_split(
        frame, test_size=0.30, random_state=seed, stratify=stratify
    )
    stratify_temp = temp_frame["label"] if temp_frame["label"].nunique() > 1 else None
    val_frame, test_frame = train_test_split(
        temp_frame, test_size=0.50, random_state=seed, stratify=stratify_temp
    )
    return train_frame.reset_index(drop=True), val_frame.reset_index(drop=True), test_frame.reset_index(drop=True)


# Feature Processing

def coerce_numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    coerced = frame[TABULAR_NUMERIC_COLUMNS].copy()
    for col in coerced.columns:
        if not pd.api.types.is_numeric_dtype(coerced[col]):
            coerced[col] = coerced[col].apply(parse_numeric)
    return coerced


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


def build_tabular_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_tax_len: int,
    reference_cat_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Builds and scales tabular numerical/categorical matrices and taxonomy tokens."""
    num_train = coerce_numeric_features(train_df.reindex(columns=TABULAR_NUMERIC_COLUMNS))
    num_val = coerce_numeric_features(val_df.reindex(columns=TABULAR_NUMERIC_COLUMNS))
    num_test = coerce_numeric_features(test_df.reindex(columns=TABULAR_NUMERIC_COLUMNS))

    # Structural columns first: a NaN here means "no fold found", which is a
    # value, not a gap. Fill it with the sentinel before medians are applied.
    for column_name in STRUCTURAL_SENTINEL_COLUMNS:
        for frame in (num_train, num_val, num_test):
            if column_name in frame.columns:
                frame[column_name] = frame[column_name].fillna(STRUCTURAL_SENTINEL_VALUE)

    medians = num_train.median(numeric_only=True)
    num_train.fillna(medians, inplace=True)
    num_val.fillna(medians, inplace=True)
    num_test.fillna(medians, inplace=True)

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(num_train)
    scaled_val = scaler.transform(num_val)
    scaled_test = scaler.transform(num_test)

    cat_train = pd.get_dummies(train_df[TABULAR_CATEGORICAL_COLUMNS].fillna("missing").astype(str))
    cat_cols = reference_cat_cols if reference_cat_cols is not None else list(cat_train.columns)
    cat_train = cat_train.reindex(columns=cat_cols, fill_value=0)
    cat_val = pd.get_dummies(val_df[TABULAR_CATEGORICAL_COLUMNS].fillna("missing").astype(str)).reindex(columns=cat_cols, fill_value=0)
    cat_test = pd.get_dummies(test_df[TABULAR_CATEGORICAL_COLUMNS].fillna("missing").astype(str)).reindex(columns=cat_cols, fill_value=0)

    train_tab_matrix = np.hstack([scaled_train, cat_train.to_numpy(dtype=np.float32)]).astype(np.float32)
    val_tab_matrix = np.hstack([scaled_val, cat_val.to_numpy(dtype=np.float32)]).astype(np.float32)
    test_tab_matrix = np.hstack([scaled_test, cat_test.to_numpy(dtype=np.float32)]).astype(np.float32)

    if "ncbi_taxonomy_id" in train_df.columns:
        tax_vocab = build_taxonomy_vocab(train_df["ncbi_taxonomy_id"])
    else:
        tax_vocab = {"<PAD>": 0, "<UNK>": 1}

    tax_train_matrix = tokenize_and_pad_taxonomy(train_df["ncbi_taxonomy_id"], tax_vocab, max_tax_len)
    tax_val_matrix = tokenize_and_pad_taxonomy(val_df["ncbi_taxonomy_id"], tax_vocab, max_tax_len)
    tax_test_matrix = tokenize_and_pad_taxonomy(test_df["ncbi_taxonomy_id"], tax_vocab, max_tax_len)

    return train_tab_matrix, val_tab_matrix, test_tab_matrix, tax_train_matrix, tax_val_matrix, tax_test_matrix, tax_vocab


def build_sequence_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    reference_max_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Encodes sequences into 3D tensors with ALL-ZERO padding and returns true lengths.
    """
    train_clean = [clean_sequence(s) for s in train_df["cleaned_primer"]]
    val_clean = [clean_sequence(s) for s in val_df["cleaned_primer"]]
    test_clean = [clean_sequence(s) for s in test_df["cleaned_primer"]]

    max_length = reference_max_length if reference_max_length is not None else max((len(s) for s in train_clean), default=1)
    max_length = max(max_length, 1)

    def encode(clean_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
        seqs = np.stack([sequence_to_tensor(s, max_length) for s in clean_list])
        lengths = np.array([min(len(s), max_length) for s in clean_list], dtype=np.int64)
        return seqs, lengths

    train_seq, train_len = encode(train_clean)
    val_seq, val_len = encode(val_clean)
    test_seq, test_len = encode(test_clean)

    return train_seq, val_seq, test_seq, train_len, val_len, test_len, int(max_length)


# Orchestration & Storage

def save_split_bundle(
    output_dir: Path, split_name: str, frame: pd.DataFrame,
    tabular: np.ndarray, taxonomy_tokens: np.ndarray, sequence: np.ndarray, sequence_lengths: np.ndarray
) -> None:
    """Packages all matrices and labels into a PyTorch dictionary and saves to disk."""
    bundle = {
        "tabular": torch.tensor(tabular, dtype=torch.float32),
        "taxonomy_tokens": torch.tensor(taxonomy_tokens, dtype=torch.long),
        "primer_sequence": torch.tensor(sequence, dtype=torch.float32),
        "sequence_lengths": torch.tensor(sequence_lengths, dtype=torch.long),
        "labels": torch.tensor(frame["label"].to_numpy(dtype=np.int64), dtype=torch.long),
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
    frame = prepare_frame(frame, label=0)
    train_df, val_df, test_df = split_labeled_frame(frame, seed=seed)

    def get_max_tax_len(df: pd.DataFrame) -> int:
        if "ncbi_taxonomy_id" not in df.columns:
            return 0
        return int(df["ncbi_taxonomy_id"].fillna("").apply(lambda x: len(parse_taxonomy_list(x))).max())

    if reference_schema is not None:
        max_tax_len = reference_schema["max_tax_len"]
    else:
        max_tax_len = max(get_max_tax_len(train_df), 1)   # train only
    print(f"Using max taxonomy padding length: {max_tax_len}")

    train_tab, val_tab, test_tab, train_tax, val_tax, test_tax, tax_vocab = build_tabular_matrices(
        train_df, val_df, test_df, max_tax_len, reference_cat_cols=(reference_schema["cat_cols"] if reference_schema else None)
    )
    train_seq, val_seq, test_seq, train_len, val_len, test_len, max_seq_len = build_sequence_matrices(
        train_df,
        val_df,
        test_df,
        reference_max_length=(reference_schema["sequence_length"] if reference_schema else None),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_split_bundle(output_dir, "train", train_df, train_tab, train_tax, train_seq, train_len)
    save_split_bundle(output_dir, "val", val_df, val_tab, val_tax, val_seq, val_len)
    save_split_bundle(output_dir, "test", test_df, test_tab, test_tax, test_seq, test_len)

    with (output_dir / "taxonomy_vocab.json").open("w", encoding="utf-8") as handle:
        json.dump(tax_vocab, handle, indent=2)

    metadata = {
        "seed": seed,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "taxonomy_vocab_size": len(tax_vocab),
        "max_taxonomy_padding_length": max_tax_len,
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
