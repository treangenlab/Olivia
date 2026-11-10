"""The eight primer properties the density head models, and their binning.

Length, GC, Tm, complexity, homopolymer run and 3' GC clamp come from the
sequence; hairpin and homodimer dG come from Head A's prediction (from NUPACK
during training). install_dg_properties() sets the active list.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from .sequence import (
    BASES,
    DEGENERATE_ENUM_CAP,
    degenerate_options,
    expand_degenerate,
    is_degenerate,
    iupac_options,
)
from .data import (
    DG_CACHE_FIELDS,
    NupackCache,
    SplitBundle,
    bundle_sequences,
)
from .targets import (
    DG_BIN_WIDTHS,
    DG_STRUCTURE_TARGETS,
    HEAD_C_DG_INDICES,
    SEQUENCE_RULE_SPEC,
)

# Representative value of a "not found" class (Binner with has_not_found);
# excluded from every average.
_NOT_FOUND_CENTER = -99.0


def max_poly_x(seq: str) -> int:
    """Longest run of one repeated base."""
    if not seq:
        return 0
    return max(len(m.group(0)) for m in re.finditer(r"(.)\1*", seq))


@lru_cache(maxsize=1 << 20)
def expected_max_poly_x(seq: str, cap: int = DEGENERATE_ENUM_CAP) -> float:
    """E[longest run of one repeated base] over the degenerate expansion."""
    if not is_degenerate(seq):
        return float(max_poly_x(seq))
    resolutions = expand_degenerate(seq, cap)
    return sum(max_poly_x(s) for s in resolutions) / len(resolutions)


# `complexity` is Olivar's measure, ported rather than reinvented, so the two agree on
# what the word means.
def sequence_complexity(seq: str) -> float:
    """Olivar's sequence complexity, ported from olivar.basic.get_complexity."""
    n = len(seq)
    if n < 3:
        return 0.0

    hash1: Counter = Counter()
    hash2: Counter = Counter()
    hash3: Counter = Counter()
    for j in range(n - 2):
        hash1[seq[j]] += 1
        hash2[seq[j:j + 2]] += 1
        hash3[seq[j:j + 3]] += 1
    hash1[seq[-2]] += 1
    hash1[seq[-1]] += 1
    hash2[seq[-2:]] += 1

    def entropy(counts: Counter, total: int) -> float:
        return -sum((c / total) * math.log2(c / total) for c in counts.values())

    se1 = entropy(hash1, n)
    se2 = entropy(hash2, n - 1)
    se3 = entropy(hash3, n - 2)
    return float(min(se1 / 2, se2 / 4, se3 / 6))


# The degenerate-aware counterparts.
def _p_gc(base: str) -> float:
    """P(this position is G or C) under a uniform draw from the base's code."""
    options = degenerate_options(base)
    return sum(1 for o in options if o in "GC") / len(options)


def expected_gc_content(seq: str) -> float:
    """E[GC%] over the degenerate expansion."""
    if not seq:
        return 0.0
    return 100.0 * sum(_p_gc(c) for c in seq) / len(seq)


def expected_gc_clamp_3p(seq: str) -> float:
    """E[G/C count in the last five bases]. Exact, same argument as GC%."""
    return sum(_p_gc(c) for c in seq[-5:])


@lru_cache(maxsize=1 << 20)
def expected_sequence_complexity(seq: str,
                                 cap: int = DEGENERATE_ENUM_CAP) -> float:
    """E[complexity] over the degenerate expansion."""
    if not is_degenerate(seq):
        return sequence_complexity(seq)
    resolutions = expand_degenerate(seq, cap)
    return sum(sequence_complexity(s) for s in resolutions) / len(resolutions)


@dataclass
class PropertySpec:
    """One coordinate of the joint distribution."""
    name: str
    kind: str
    violates: object
    width: float = 1.0
    anchors: tuple = ()
    lo_q: float = 0.005
    hi_q: float = 0.995
    int_lo: int = 0
    int_hi: int = 1
    has_not_found: bool = False
    not_found_column: str | None = None


# Discretisation: each property is binned so every conditional is a softmax.
class Binner:
    """Bin edges plus the class-index / class-centre / violating-class mapping
    for one property."""

    def __init__(self, spec: PropertySpec, edges: np.ndarray):
        self.spec = spec
        self.edges = edges.astype(np.float64)
        self.off = 1 if spec.has_not_found else 0
        self.n_classes = len(self.edges) + 1 + self.off

    # -- construction ------------------------------------------------------
    @staticmethod
    def fit(spec: PropertySpec, values: np.ndarray, found: np.ndarray) -> "Binner":
        if spec.kind == "integer":
            # One class per integer in [int_lo, int_hi]; half-integer edges put
            # each integer squarely inside its own bin under np.digitize.
            edges = np.arange(spec.int_lo + 0.5, spec.int_hi + 0.5, 1.0)
            return Binner(spec, edges)

        sel = values[found > 0.5]
        if sel.size == 0:
            return Binner(spec, np.array([0.0, spec.width], dtype=np.float64))

        lo = float(np.quantile(sel, spec.lo_q))
        hi = float(np.quantile(sel, spec.hi_q))
        anchor = float(spec.anchors[0]) if spec.anchors else float(np.round(lo / spec.width) * spec.width)

        k_lo = int(math.floor((lo - anchor) / spec.width))
        k_hi = int(math.ceil((hi - anchor) / spec.width))
        # Anchors (limits) must lie strictly inside the grid.
        for a in spec.anchors:
            k = (float(a) - anchor) / spec.width
            if abs(k - round(k)) > 1e-6:
                raise ValueError(
                    f"{spec.name}: anchor {a} is not a multiple of width {spec.width} from {anchor}; "
                    f"pick a width dividing the gap between anchors so every limit lands on an edge"
                )
            k_lo, k_hi = min(k_lo, int(round(k))), max(k_hi, int(round(k)))
        edges = anchor + spec.width * np.arange(k_lo, k_hi + 1, dtype=np.float64)
        return Binner(spec, edges)

    # -- mapping -----------------------------------------------------------
    def to_class(self, values: np.ndarray, found: np.ndarray) -> np.ndarray:
        v = values
        if self.spec.kind == "integer":
            v = np.clip(v, self.spec.int_lo, self.spec.int_hi)
        cls = np.digitize(v, self.edges) + self.off
        if self.off:
            cls[found <= 0.5] = 0
        return cls.astype(np.int64)

    def centers(self) -> np.ndarray:
        """Representative value per class. The not-found class, where present,
        gets the sentinel and callers must exclude it rather than average it."""
        c = np.empty(self.n_classes, dtype=np.float64)
        w = self.spec.width if self.spec.kind == "continuous" else 1.0
        if self.off:
            c[0] = _NOT_FOUND_CENTER
        c[self.off] = self.edges[0] - w / 2.0
        for k in range(self.off + 1, self.n_classes):
            lower = self.edges[k - self.off - 1]
            upper = self.edges[k - self.off] if k - self.off < len(self.edges) else lower + w
            c[k] = (lower + upper) / 2.0
        return c

    def violating_mask(self) -> np.ndarray:
        """Classes that count as a violation. The not-found class never
        does -- a primer that does not fold has not violated a folding limit."""
        mask = np.array([bool(self.spec.violates(v)) for v in self.centers()], dtype=bool)
        if self.off:
            mask[0] = False
        return mask

    def label(self, k: int) -> str:
        c = self.centers()
        if self.off and k == 0:
            return "none"
        if self.spec.kind == "integer":
            return f"{int(round(c[k]))}"
        # Enough decimals to keep adjacent bin labels distinct.
        decimals = max(0, math.ceil(-math.log10(self.spec.width))) + 1
        return f"{c[k]:.{decimals}f}"


def to_class_matrix(values: np.ndarray, found: np.ndarray, binners: list[Binner]) -> torch.Tensor:
    out = np.zeros(values.shape, dtype=np.int64)
    for j, b in enumerate(binners):
        out[:, j] = b.to_class(values[:, j], found[:, j])
    return torch.from_numpy(out)


def binners_from_checkpoint(ckpt: dict) -> list[Binner]:
    """Rebuild the training-corpus discretisation from a saved checkpoint."""
    names = list(ckpt.get("property_names", []))
    if names != PROPERTY_NAMES:
        raise ValueError(
            f"checkpoint property vector {names} does not match this module's {PROPERTY_NAMES}; "
            "PROPERTIES was edited after the checkpoint was written -- retrain rather than score."
        )
    binners = [Binner(spec, np.asarray(e, dtype=np.float64))
               for spec, e in zip(PROPERTIES, ckpt["property_edges"])]
    saved, rebuilt = list(ckpt["property_class_counts"]), [b.n_classes for b in binners]
    if saved != rebuilt:
        raise ValueError(
            f"checkpoint class counts {saved} != rebuilt {rebuilt}; the bin layout has changed "
            "(a PropertySpec width, range or not-found flag was edited) -- retrain."
        )
    return binners


# Duplex Tm, SantaLucia (1998) nearest-neighbour parameters:
# Tm = 1000 dH / (dS' + R ln(Ct / f)) - 273.15, with salt-corrected dS'.
# Degenerate primers average dH and dS over their bases (TM_DEGENERATE).

# Stacking enthalpies (kcal/mol) and entropies (cal/(mol K)), 5'->3' on the top strand.
_TM_NN_DH = {"AA": -7.9, "AT": -7.2, "TA": -7.2, "CA": -8.5, "GT": -8.4,
             "CT": -7.8, "GA": -8.2, "CG": -10.6, "GC": -9.8, "GG": -8.0}


_TM_NN_DS = {"AA": -22.2, "AT": -20.4, "TA": -21.3, "CA": -22.7, "GT": -22.4,
             "CT": -21.0, "GA": -22.2, "CG": -27.2, "GC": -24.4, "GG": -19.9}


# Helix initiation, paid once per end and depending on which pair closes it: an
# A-T end frays more readily than a G-C one.
_TM_INIT_DH = {"A": 2.3, "T": 2.3, "C": 0.1, "G": 0.1}


_TM_INIT_DS = {"A": 4.1, "T": 4.1, "C": -2.8, "G": -2.8}


_TM_SYMMETRY_DS = -1.4


# R as used by the published nearest-neighbour tables.
_GAS_CONSTANT = 1.987


_COMPLEMENT = str.maketrans("ACGT", "TGCA")


# Buffer and oligo concentrations the Tm is quoted at: a standard PCR mix. Set
# from the command line by train.main() and recorded in the checkpoint.
TM_CONDITIONS = {"mv_conc_mm": 50.0, "dv_conc_mm": 1.5,
                 "dntp_conc_mm": 0.6, "dna_conc_nm": 50.0}


# IUPAC handling for Tm: 'expected' over the bases a code denotes, or 'first'.
TM_DEGENERATE = "expected"


TM_DEGENERATE_MODES = ("expected", "first")


def _expand(table: dict[str, float]) -> dict[str, float]:
    out = dict(table)
    for stack, value in table.items():
        out.setdefault(stack.translate(_COMPLEMENT)[::-1], value)
    return out


TM_NN_DH = _expand(_TM_NN_DH)


TM_NN_DS = _expand(_TM_NN_DS)


def effective_monovalent(mv_conc_mm: float, dv_conc_mm: float, dntp_conc_mm: float) -> float:
    """Divalent cations expressed as the monovalent concentration (mM) that
    would stabilise a duplex equally. dNTPs chelate Mg2+ roughly 1:1, so only
    the excess counts; with no divalent present there is nothing to chelate."""
    if dv_conc_mm <= 0.0:
        dntp_conc_mm = 0.0
    dv_conc_mm = max(dv_conc_mm, dntp_conc_mm)
    return mv_conc_mm + 120.0 * math.sqrt(dv_conc_mm - dntp_conc_mm)


def _has_ambiguity(seq: str) -> bool:
    return any(b not in BASES for b in seq.upper().replace("U", "T"))


def _nn_thermodynamics(options: list[str]) -> tuple[float, float]:
    """(dH kcal/mol, dS cal/(mol K)) averaged over the pool `options` denotes."""
    dh = sum(_TM_INIT_DH[b] for b in options[0]) / len(options[0])
    ds = sum(_TM_INIT_DS[b] for b in options[0]) / len(options[0])
    dh += sum(_TM_INIT_DH[b] for b in options[-1]) / len(options[-1])
    ds += sum(_TM_INIT_DS[b] for b in options[-1]) / len(options[-1])

    for left, right in zip(options, options[1:]):
        n = len(left) * len(right)
        dh += sum(TM_NN_DH[x + y] for x in left for y in right) / n
        ds += sum(TM_NN_DS[x + y] for x in left for y in right) / n
    return dh, ds


def nn_duplex_tm(seq: str, conditions: dict[str, float] | None = None,
                 degenerate: str | None = None) -> float:
    """Tm (degC) of `seq` against its perfect complement."""
    cond = conditions or TM_CONDITIONS
    mode = degenerate or TM_DEGENERATE
    if mode not in TM_DEGENERATE_MODES:
        raise ValueError(f"degenerate must be one of {TM_DEGENERATE_MODES}, got {mode!r}")

    options = iupac_options(seq)
    if options is None or len(options) < 2:
        return float("nan")
    if mode == "first":
        options = [opts[0] for opts in options]

    dh, ds = _nn_thermodynamics(options)

    # Only a primer with no ambiguity anywhere anneals to a copy of itself as a
    # whole pool; a degenerate one is a mixture, and its Ct is not its own.
    resolved = "".join(opts[0] for opts in options)
    self_complementary = (all(len(opts) == 1 for opts in options)
                          and resolved == resolved.translate(_COMPLEMENT)[::-1])
    if self_complementary:
        ds += _TM_SYMMETRY_DS

    na = effective_monovalent(cond["mv_conc_mm"], cond["dv_conc_mm"],
                              cond["dntp_conc_mm"]) / 1000.0
    ds += 0.368 * (len(options) - 1) * math.log(na)

    ct = cond["dna_conc_nm"] / 1e9 / (1.0 if self_complementary else 4.0)
    return 1000.0 * dh / (ds + _GAS_CONSTANT * math.log(ct)) - 273.15


def nn_duplex_tm_array(seqs: list[str], conditions: dict[str, float] | None = None,
                       degenerate: str | None = None) -> np.ndarray:
    cond = conditions or TM_CONDITIONS
    mode = degenerate or TM_DEGENERATE
    return np.array([nn_duplex_tm(s, cond, mode) for s in seqs], dtype=np.float32)


def tm_summary(seqs: list[str], tm: np.ndarray) -> dict[str, float]:
    """Where the computed Tm sits, split by whether degeneracy was averaged over."""
    degenerate = np.array([_has_ambiguity(s) for s in seqs])
    out = {"n": int(tm.size), "n_degenerate": int(degenerate.sum()),
           "mean": float(np.nanmean(tm)),
           "lo": float(np.nanmin(tm)), "hi": float(np.nanmax(tm))}
    for label, sel in (("concrete", ~degenerate), ("degenerate", degenerate)):
        out[f"mean_{label}"] = float(np.nanmean(tm[sel])) if sel.any() else float("nan")
    return out


# The property vector
def dg_properties(limits: list[float],
                  bounds: list[tuple[float, float]] | None = None) -> list[PropertySpec]:
    """Head C's eight coordinates: six from the sequence, two structural."""
    structural = []
    for spec_j, j in enumerate(HEAD_C_DG_INDICES):
        limit = limits[j]
        anchors = (limit,) if bounds is None else (limit, bounds[j][0], bounds[j][1])
        structural.append(
            # `v <= limit`, not `v >= limit`: stability is the violation.
            PropertySpec(DG_STRUCTURE_TARGETS[j], "continuous",
                         lambda v, k=limit: v <= k,
                         width=DG_BIN_WIDTHS[j], anchors=anchors))
    return [
        PropertySpec("length", "integer",
                     lambda v: v < SEQUENCE_RULE_SPEC["length_min"] or v > SEQUENCE_RULE_SPEC["length_max"],
                     int_lo=10, int_hi=45),
        PropertySpec("gc", "continuous",
                     lambda v: v < SEQUENCE_RULE_SPEC["gc_min"] or v > SEQUENCE_RULE_SPEC["gc_max"],
                     width=2.5, anchors=(SEQUENCE_RULE_SPEC["gc_min"], SEQUENCE_RULE_SPEC["gc_max"])),
        PropertySpec("tm", "continuous",
                     lambda v: v < SEQUENCE_RULE_SPEC["tm_min"] or v > SEQUENCE_RULE_SPEC["tm_max"],
                     width=0.5, anchors=(SEQUENCE_RULE_SPEC["tm_min"], SEQUENCE_RULE_SPEC["tm_max"])),
        PropertySpec("complexity", "continuous", lambda v: False, width=0.02),
        PropertySpec("poly_x", "integer", lambda v: v > SEQUENCE_RULE_SPEC["poly_x_max"], int_lo=1, int_hi=12),
        PropertySpec("gc_clamp", "integer", lambda v: False, int_lo=0, int_hi=5),
        *structural,
    ]


def install_dg_properties(limits: list[float],
                          bounds: list[tuple[float, float]] | None = None) -> list[str]:
    """Make the dG property list (with these limits) the active PROPERTIES."""
    global PROPERTIES, PROPERTY_NAMES
    PROPERTIES = dg_properties(limits, bounds)
    PROPERTY_NAMES = [p.name for p in PROPERTIES]
    return PROPERTY_NAMES


# How many of Head C's coordinates are computed from the sequence string.
SEQ_PROPERTY_COUNT = 6
N_PROPERTIES = SEQ_PROPERTY_COUNT + len(HEAD_C_DG_INDICES)

# The active property list; empty until install_dg_properties() runs.
PROPERTIES: list[PropertySpec] = []
PROPERTY_NAMES: list[str] = []


def sequence_property_values(bundle: SplitBundle) -> tuple[np.ndarray, np.ndarray]:
    """(N, 6) sequence-derived property values and (N, 6) all-ones found flags."""
    seqs = bundle_sequences(bundle)
    props = PROPERTIES[:SEQ_PROPERTY_COUNT]
    values = np.zeros((len(seqs), len(props)), dtype=np.float32)

    tm = nn_duplex_tm_array(seqs)
    unusable = int(np.isnan(tm).sum())
    if unusable:
        # Warn when a Tm cannot be computed (filled with tm_min).
        print(f"    warning: {unusable} of {len(seqs)} sequence(s) have no resolvable "
              f"nearest-neighbour Tm; filling with tm_min = {SEQUENCE_RULE_SPEC['tm_min']} degC")
    tm = np.nan_to_num(tm, nan=float(SEQUENCE_RULE_SPEC["tm_min"]))

    for j, spec in enumerate(props):
        if spec.name == "length":
            values[:, j] = [len(s) for s in seqs]
        elif spec.name == "gc":
            values[:, j] = [expected_gc_content(s) for s in seqs]
        elif spec.name == "tm":
            values[:, j] = tm
        elif spec.name == "complexity":
            values[:, j] = [expected_sequence_complexity(s) for s in seqs]
        elif spec.name == "poly_x":
            values[:, j] = [expected_max_poly_x(s) for s in seqs]
        elif spec.name == "gc_clamp":
            values[:, j] = [expected_gc_clamp_3p(s) for s in seqs]
        else:
            raise KeyError(f"no source for property {spec.name!r}")

    return values, np.ones_like(values)


def extract_property_values(bundle: SplitBundle,
                            cache: NupackCache) -> tuple[np.ndarray, np.ndarray]:
    """(N, 8) property values and (N, 8) all-ones found flags, dG read from NUPACK."""
    seq_values, _ = sequence_property_values(bundle)
    dg = cache.energies_for(bundle_sequences(bundle), DG_CACHE_FIELDS).astype(np.float32)
    # Head C's structural columns, in HEAD_C_DG_INDICES order.
    values = np.concatenate([seq_values, dg[:, list(HEAD_C_DG_INDICES)]], axis=1)
    return values, np.ones_like(values)


def fit_binners(values: np.ndarray, found: np.ndarray) -> list[Binner]:
    return [Binner.fit(spec, values[:, j], found[:, j])
            for j, spec in enumerate(PROPERTIES)]
