"""Primer strings: cleaning, IUPAC degeneracy and the encoder's input tensor.

A primer of length n <= L is encoded as an (L, 4) matrix over A, C, G, T. A
degenerate base is a uniform distribution over the bases it stands for
(W -> A/T at 0.5 each, N -> 0.25 each), and padding positions are all-zero, so
a position is real exactly when its row sums to 1.
"""
from __future__ import annotations

import itertools
import random

import numpy as np
import pandas as pd

BASES = "ACGT"
_BASE_INDEX = {b: i for i, b in enumerate(BASES)}

IUPAC_BASES = {
    "A": "A", "C": "C", "G": "G", "T": "T", "U": "T",
    "R": "AG", "Y": "CT", "S": "GC", "W": "AT", "K": "GT", "M": "AC",
    "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT",
}

# Unknown symbols and gaps are treated as fully ambiguous, so they stay
# distinguishable from padding.
_ACGT = ("A", "C", "G", "T")


def clean_sequence(value) -> str:
    """Uppercase and strip a primer string; NaN/None become ''."""
    if value is None or pd.isna(value):
        return ""
    return str(value).upper().strip()


def base_distribution(char: str) -> np.ndarray:
    """One IUPAC symbol as a probability distribution over A, C, G, T."""
    vec = np.zeros(len(BASES), dtype=np.float32)
    for b in IUPAC_BASES.get(char, "ACGT"):
        vec[_BASE_INDEX[b]] = 1.0
    return vec / vec.sum()


def sequence_to_tensor(sequence: str, max_length: int) -> np.ndarray:
    """(max_length, 4) encoding of a cleaned primer string, zero-padded."""
    encoded = np.zeros((max_length, len(BASES)), dtype=np.float32)
    for pos, char in enumerate(sequence[:max_length]):
        encoded[pos] = base_distribution(char)
    return encoded


def concrete(seq: str) -> str | None:
    """Resolve IUPAC degeneracy to concrete ACGT (first listed option), for
    tools that will not fold ambiguous bases. None if a letter is not IUPAC."""
    out = []
    for base in seq.upper().replace("U", "T"):
        opts = IUPAC_BASES.get(base)
        if opts is None:
            return None
        out.append(opts[0])
    return "".join(out)


def iupac_options(seq: str) -> list[str] | None:
    """Per position, the bases the written code stands for. None if any letter
    is not IUPAC -- the same all-or-nothing rule concrete applies."""
    options = []
    for base in seq.upper().replace("U", "T"):
        opts = IUPAC_BASES.get(base)
        if opts is None:
            return None
        options.append(opts)
    return options


# Degenerate primers: a property is averaged over the primer's expansions, enumerated
# up to DEGENERATE_ENUM_CAP and sampled (seeded by the sequence) beyond it.
DEGENERATE_ENUM_CAP = 64


def degenerate_options(base: str) -> tuple[str, ...]:
    return IUPAC_BASES.get(base, _ACGT)


def is_degenerate(seq: str) -> bool:
    return any(c not in "ACGT" for c in seq)


def degenerate_expansion_size(seq: str) -> int:
    n = 1
    for c in seq:
        n *= len(degenerate_options(c))
    return n


def expand_degenerate(seq: str, cap: int = DEGENERATE_ENUM_CAP) -> list[str]:
    """The concrete resolutions of `seq`, or `cap` uniform draws from them."""
    if not is_degenerate(seq):
        return [seq]
    if degenerate_expansion_size(seq) <= cap:
        return ["".join(c) for c in itertools.product(*(degenerate_options(b) for b in seq))]
    rng = random.Random(seq)
    return ["".join(rng.choice(degenerate_options(b)) for b in seq) for _ in range(cap)]
