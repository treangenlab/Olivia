"""Step 1 of dataset preparation: add calculated primer properties to a CSV.

    python -m olivia_encoder.prepare_primers --input primers.csv --output
    primers_calculated.csv

Needs a `cleaned_primer` column; adds Tm, length, GC, dG and primer3
hairpin/homodimer results.
"""
from __future__ import annotations

import argparse
import itertools
import math
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .sequence import concrete, iupac_options

try:
    import primer3
except ImportError:
    primer3 = None


IUPAC_GC_FRACTION = {
    "A": 0.0,
    "C": 1.0,
    "G": 1.0,
    "T": 0.0,
    "U": 0.0,
    "R": 0.5,
    "Y": 0.5,
    "S": 1.0,
    "W": 0.0,
    "K": 0.5,
    "M": 0.5,
    "B": 2.0 / 3.0,
    "D": 1.0 / 3.0,
    "H": 1.0 / 3.0,
    "V": 2.0 / 3.0,
    "N": 0.5,
}

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

BASE_TO_INDEX = {
    "A": 0,
    "T": 1,
    "C": 2,
    "G": 3,
}


PARA_H = (
    (-7.6, -7.2, -8.4, -7.8),
    (-7.2, -7.6, -8.2, -8.5),
    (-8.5, -7.8, -8.0, -10.6),
    (-8.2, -8.4, -9.8, -8.0),
)

PARA_S = (
    (-21.3, -20.4, -22.4, -21.0),
    (-21.3, -21.3, -22.2, -22.7),
    (-22.7, -21.0, -19.9, -27.2),
    (-22.2, -22.4, -24.4, -19.9),
)

# Matches olivar.design.primer_generator's default salinity (0.18 M); it
# corrects the entropy table by +0.368*ln(salinity) before building dG.
DEFAULT_SALINITY = 0.18
_SALINITY_OFFSET = 0.368 * math.log(DEFAULT_SALINITY)
PARA_S_ADJUSTED = tuple(tuple(s + _SALINITY_OFFSET for s in row) for row in PARA_S)


def normalize_primer(value: object) -> str:
    if pd.isna(value):
        return ""

    return "".join(str(value).split()).upper().replace("U", "T")


def parse_numeric(value: object) -> float:
    if pd.isna(value):
        return math.nan

    match = FLOAT_RE.search(str(value).replace(",", ""))
    if not match:
        return math.nan

    return float(match.group())


def choose_primer_column(df: pd.DataFrame) -> str:
    for candidate in ("cleaned_primer",):
        if candidate in df.columns:
            return candidate

    raise KeyError(
        "Could not find a primer column. Expected 'cleaned_primer'. "
        f"Available columns: {', '.join(df.columns)}"
    )


def calculate_gc_fraction(sequence: str) -> float:
    if not sequence:
        return math.nan

    gc_fraction = 0.0
    for base in sequence:
        gc_fraction += IUPAC_GC_FRACTION.get(base, 0.0)

    return gc_fraction / len(sequence)


def calculate_delta_g(sequence: str, temperature: float) -> float:
    """Nearest-neighbour dG (kcal/mol), matching Olivar's primer_generator."""
    if not sequence:
        return math.nan
    if math.isnan(temperature):
        return math.nan

    base_options = iupac_options(sequence)
    if base_options is None or len(base_options) < 2:
        return math.nan

    def calculate_concrete_delta_g(concrete_sequence: str) -> float:
        indices = [BASE_TO_INDEX[base] for base in concrete_sequence]
        dg_init = 0.2 - (temperature + 273.15) * (-5.7) / 1000
        total = dg_init
        for index in range(len(indices) - 1):
            h = PARA_H[indices[index]][indices[index + 1]]
            s = PARA_S_ADJUSTED[indices[index]][indices[index + 1]]
            total += h - (temperature + 273.15) * s / 1000

        return float(total)

    concrete_sequences = ("".join(bases) for bases in itertools.product(*base_options))
    total = 0.0
    count = 0
    for concrete_sequence in concrete_sequences:
        total += calculate_concrete_delta_g(concrete_sequence)
        count += 1

    return float(total / count) if count else math.nan


# Temperature (degC) at which the self-structure search is run.
STRUCTURE_SEARCH_TEMP_C = 37.0

# Sentinel written to *_tm when primer3 finds no fold.
NO_STRUCTURE_TM = -99.0

STRUCTURAL_COLUMNS = (
    "hairpin_found", "hairpin_dg", "hairpin_tm",
    "homodimer_found", "homodimer_dg", "homodimer_tm",
)


def calculate_structural_dg(sequence: str, temperature: float = STRUCTURE_SEARCH_TEMP_C) -> tuple[float, ...]:
    """Hairpin and homodimer folding free energy (kcal/mol) via primer3's nearest-neighbour
    self-structure search.
    """
    unknown = (math.nan,) * len(STRUCTURAL_COLUMNS)

    if primer3 is None or not sequence:
        return unknown

    if not (2 <= len(sequence) <= 60):
        return unknown

    # representative resolution; primer3 needs concrete ACGT
    concrete_seq = concrete(sequence)
    if concrete_seq is None:
        return unknown

    hairpin   = primer3.calc_hairpin(concrete_seq, temp_c=temperature)
    homodimer = primer3.calc_homodimer(concrete_seq, temp_c=temperature)
    return (
        float(hairpin.structure_found),
        hairpin.dg / 1000.0 if hairpin.structure_found else 0.0,
        hairpin.tm if hairpin.structure_found else NO_STRUCTURE_TM,
        float(homodimer.structure_found),
        homodimer.dg / 1000.0 if homodimer.structure_found else 0.0,
        homodimer.tm if homodimer.structure_found else NO_STRUCTURE_TM,
    )


def calculate_annealing_temp(sequence: str) -> float:
    """Tm (degC) from primer3's nearest-neighbour model; Wallace rule if primer3 is
    missing.
    """
    if not sequence:
        return math.nan

    if primer3 is not None:
        # representative resolution; primer3 needs concrete ACGT
        concrete_seq = concrete(sequence)
        if concrete_seq is None or len(concrete_seq) < 2:
            return math.nan
        return float(primer3.calc_tm(concrete_seq))

    gc_fraction = calculate_gc_fraction(sequence)
    if math.isnan(gc_fraction):
        return math.nan

    gc_count = gc_fraction * len(sequence)
    at_count = len(sequence) - gc_count
    return 2.0 * at_count + 4.0 * gc_count


def add_calculated_info_batch(df: pd.DataFrame, primer_column: str) -> pd.DataFrame:
    for column_name in ("annealing_temp", "Mg_concentration", "primer_concentration", "amplicon_size"):
        if column_name in df.columns:
            df[column_name] = df[column_name].map(parse_numeric)

    normalized_primers = df[primer_column].apply(normalize_primer)
    df["calculated_annealing_temp"] = normalized_primers.apply(calculate_annealing_temp).round(2)
    df["primer_length"] = normalized_primers.str.len()
    df["gc_content"] = (normalized_primers.apply(calculate_gc_fraction) * 100).round(2)

    if "annealing_temp" in df.columns:
        temperatures = df["annealing_temp"].fillna(60.0)
    else:
        temperatures = pd.Series(60.0, index=df.index)

    df["delta_g_kcal"] = [
        round(calculate_delta_g(sequence, float(temperature)), 2)
        for sequence, temperature in zip(normalized_primers, temperatures)
    ]

    # Fixed search temperature: it only gates whether primer3 reports a fold.
    structural = [
        calculate_structural_dg(sequence, STRUCTURE_SEARCH_TEMP_C)
        for sequence in normalized_primers
    ]
    for position, column_name in enumerate(STRUCTURAL_COLUMNS):
        values = [row[position] for row in structural]
        df[column_name] = [
            value if math.isnan(value) else round(value, 2) for value in values
        ]

    df["source_count"] = ""   # placeholder column expected by build_dataset

    return df


def add_calculated_info(input_path: Path, output_path: Path, batch_size: int = 5000) -> int:
    total_rows = sum(1 for _ in open(input_path)) - 1
    reader = pd.read_csv(input_path, chunksize=batch_size)

    rows_written = 0
    with tqdm(total=total_rows, unit="row", desc=output_path.name) as progress:
        for batch_index, chunk in enumerate(reader):
            primer_column = choose_primer_column(chunk)
            chunk = add_calculated_info_batch(chunk, primer_column)

            chunk.to_csv(output_path, index=False, mode="w" if batch_index == 0 else "a", header=batch_index == 0)

            rows_written += len(chunk)
            progress.update(len(chunk))

    return rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add calculated primer properties (Tm, length, GC, dG, primer3 "
            "hairpin/homodimer) to a primer CSV with a cleaned_primer column."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV file containing primer records.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV file with the calculated columns added.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Number of rows to process and write per batch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows_written = add_calculated_info(args.input, args.output, batch_size=args.batch_size)
    print(f"Saved {rows_written} rows to {args.output}")


if __name__ == "__main__":
    main()
