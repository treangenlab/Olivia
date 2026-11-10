"""Step 4 of dataset preparation: fold every unique primer with NUPACK 4.

    python -m olivia_encoder.nupack_targets --embedded-dir <dir>

Writes <dir>/nupack/nupack_cache.npz: per sequence the ensemble and MFE free
energies of the hairpin and homodimer, the MFE partner map, and the top-K
ensemble pair probabilities. Needs only numpy and nupack, so it can run in a
separate NUPACK environment.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_EMBEDDED_DIR = Path(__file__).resolve().parents[2] / "data" / "embedded" / "true_data_all_t37"

# Duplicated from sequence.py rather than imported, so this file runs as a plain
# script with no package on the path.
IUPAC_BASES = {
    "A": "A", "C": "C", "G": "G", "T": "T", "U": "T",
    "R": "AG", "Y": "CT", "S": "GC", "W": "AT", "K": "GT", "M": "AC",
    "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT",
}

# Folding conditions: 37 degC, 50 mM Na+, no Mg2+ (primer3-py's defaults).
FOLD_TEMP_C = 37.0
SODIUM_M = 0.05
MAGNESIUM_M = 0.0

FIELDS = ("hairpin_dg", "hairpin_dg_mfe", "homodimer_dg", "homodimer_dg_mfe")

# Partners kept per position from the ensemble pair matrix, and the probability floor.
PAIR_TOP_K = 4
PAIR_PROB_MIN = 1e-4

_MODEL = None


def _model():
    """One Model per worker process, built lazily so it is never pickled."""
    global _MODEL
    if _MODEL is None:
        from nupack import Model
        _MODEL = Model(material="dna", celsius=FOLD_TEMP_C,
                       sodium=SODIUM_M, magnesium=MAGNESIUM_M)
    return _MODEL


def concrete(seq: str) -> str | None:
    """Resolve IUPAC degeneracy to concrete ACGT, matching sequence.concrete."""
    out = []
    for base in seq.upper().replace("U", "T"):
        opts = IUPAC_BASES.get(base)
        if opts is None:
            return None
        out.append(opts[0])
    return "".join(out)


def _energy(value: float) -> float:
    """0.0 kcal/mol for a complex that cannot form."""
    return 0.0 if not np.isfinite(value) else float(value)


def _pair_probabilities(cseq: str, model, max_len: int, n: int
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-K ensemble pair probabilities -> (idx, prob, unpaired)."""
    from nupack import pairs as pair_matrix

    top_idx = np.full((max_len, PAIR_TOP_K), -1, dtype=np.int16)
    top_prob = np.zeros((max_len, PAIR_TOP_K), dtype=np.float32)
    unpaired = np.ones(max_len, dtype=np.float32)

    arr = np.asarray(pair_matrix([cseq], model=model).to_array(), dtype=np.float64)
    unpaired[:n] = np.diag(arr)[:n]

    off = arr[:n, :n].copy()
    np.fill_diagonal(off, 0.0)
    off[off < PAIR_PROB_MIN] = 0.0

    # Full sort keeps partners in descending probability.
    k = min(PAIR_TOP_K, n)
    order = np.argsort(-off, axis=1)[:, :k]
    vals = np.take_along_axis(off, order, axis=1)
    keep = vals > 0.0
    top_idx[:n, :k] = np.where(keep, order, -1).astype(np.int16)
    top_prob[:n, :k] = np.where(keep, vals, 0.0).astype(np.float32)
    return top_idx, top_prob, unpaired


def fold_one(seq: str, max_len: int) -> tuple[np.ndarray, ...]:
    """-> (4 free energies, MFE partner map, top-K pair idx / prob / unpaired)."""
    from nupack import mfe, pfunc

    energies = np.zeros(len(FIELDS), dtype=np.float32)
    partner = np.full(max_len, -1, dtype=np.int16)
    top_idx = np.full((max_len, PAIR_TOP_K), -1, dtype=np.int16)
    top_prob = np.zeros((max_len, PAIR_TOP_K), dtype=np.float32)
    unpaired = np.ones(max_len, dtype=np.float32)
    empty = (energies, partner, top_idx, top_prob, unpaired)

    cseq = concrete(seq)
    if cseq is None or len(cseq) < 2:
        return empty

    model = _model()
    n = min(len(cseq), max_len)

    energies[0] = _energy(pfunc([cseq], model=model)[1])
    monomer = mfe([cseq], model=model)
    energies[1] = _energy(monomer[0].energy) if monomer else 0.0
    energies[2] = _energy(pfunc([cseq, cseq], model=model)[1])
    dimer = mfe([cseq, cseq], model=model)
    energies[3] = _energy(dimer[0].energy) if dimer else 0.0

    # The ensemble pair matrix exists even when no MFE fold does.
    top_idx, top_prob, unpaired = _pair_probabilities(cseq, model, max_len, n)

    if not monomer:
        return energies, partner, top_idx, top_prob, unpaired

    # NUPACK marks an unpaired base i with p[i] == i; the training code wants -1.
    hairpin = monomer[0]
    pairs = np.asarray(hairpin.structure.pairlist(), dtype=np.int64)
    idx = np.arange(len(pairs))
    paired = pairs != idx
    partner[:n] = np.where(paired, pairs, -1)[:n].astype(np.int16)
    # A partner beyond the truncation point is not representable, so drop the
    # half-pair rather than leave a dangling index into padding.
    partner[:n][partner[:n] >= n] = -1
    return energies, partner, top_idx, top_prob, unpaired


_MAX_LEN = 0


def _init(max_len: int) -> None:
    global _MAX_LEN
    _MAX_LEN = max_len
    _model()


def _work(seq: str):
    return fold_one(seq, _MAX_LEN)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fold primers with NUPACK and cache dG / pairing targets.")
    p.add_argument("--embedded-dir", type=Path, default=DEFAULT_EMBEDDED_DIR)
    p.add_argument("--sequences", type=Path, default=None,
                   help="Input list (default: <embedded-dir>/nupack/sequences.txt).")
    p.add_argument("--out", type=Path, default=None,
                   help="Destination .npz (default: <embedded-dir>/nupack/nupack_cache.npz).")
    p.add_argument("--max-len", type=int, default=60,
                   help="Pairing maps are padded to this length.")
    p.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    p.add_argument("--limit", type=int, default=0, help="Fold only the first N sequences (smoke run).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = args.sequences or (args.embedded_dir / "nupack" / "sequences.txt")
    out = args.out or (args.embedded_dir / "nupack" / "nupack_cache.npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise SystemExit(f"{src} not found -- run nupack_dump_sequences.py in the torch env first.")

    seqs = [s.strip() for s in src.read_text(encoding="utf-8").splitlines() if s.strip()]
    if args.limit:
        seqs = seqs[:args.limit]

    import nupack
    print(f"NUPACK {nupack.__version__} | {FOLD_TEMP_C} degC, Na+ {SODIUM_M} M, Mg2+ {MAGNESIUM_M} M")
    print(f"folding {len(seqs)} unique sequences on {args.workers} workers ...")

    energies = np.zeros((len(seqs), len(FIELDS)), dtype=np.float32)
    partner = np.full((len(seqs), args.max_len), -1, dtype=np.int16)
    pair_top_idx = np.full((len(seqs), args.max_len, PAIR_TOP_K), -1, dtype=np.int16)
    pair_top_prob = np.zeros((len(seqs), args.max_len, PAIR_TOP_K), dtype=np.float32)
    pair_unpaired = np.ones((len(seqs), args.max_len), dtype=np.float32)

    def store(i, row):
        energies[i], partner[i], pair_top_idx[i], pair_top_prob[i], pair_unpaired[i] = row

    started = time.time()
    if args.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_init, initargs=(args.max_len,)) as pool:
            for i, row in enumerate(pool.imap(_work, seqs, chunksize=64)):
                store(i, row)
                if (i + 1) % 5000 == 0:
                    rate = (i + 1) / (time.time() - started)
                    print(f"    {i+1}/{len(seqs)}  {rate:.0f} seq/s  "
                          f"eta {(len(seqs)-i-1)/rate/60:.1f} min", flush=True)
    else:
        _init(args.max_len)
        for i, s in enumerate(seqs):
            store(i, fold_one(s, args.max_len))

    elapsed = time.time() - started
    np.savez_compressed(
        out,
        sequences=np.array(seqs, dtype=object),
        energies=energies,
        partner=partner,
        pair_top_idx=pair_top_idx,
        pair_top_prob=pair_top_prob,
        pair_unpaired=pair_unpaired,
        pair_top_k=np.int64(PAIR_TOP_K),
        pair_prob_min=np.float64(PAIR_PROB_MIN),
        fields=np.array(FIELDS),
        max_len=np.int64(args.max_len),
        fold_temp_c=np.float64(FOLD_TEMP_C),
        sodium_m=np.float64(SODIUM_M),
        magnesium_m=np.float64(MAGNESIUM_M),
        nupack_version=np.array(nupack.__version__),
    )

    print(f"\ndone in {elapsed/60:.1f} min -> {out}")
    paired_frac = float((partner >= 0).sum() / max((partner != -1).size, 1))
    for j, name in enumerate(FIELDS):
        v = energies[:, j]
        print(f"  {name:18s} mean {v.mean():8.3f}  min {v.min():8.3f}  max {v.max():8.3f}  "
              f"zero {float((v == 0).mean()):.4f}")
    print(f"  positions paired in the MFE hairpin: {paired_frac:.4f}")

    # How much of the corpus the MFE and ensemble targets leave unpaired.
    mfe_empty = float((partner < 0).all(axis=1).mean())
    ens_empty = float((pair_top_idx < 0).all(axis=(1, 2)).mean())
    kept = pair_top_prob.sum(axis=2)
    print(f"  sequences with an empty MFE pair list:        {mfe_empty:.4f}")
    print(f"  sequences with no ensemble partner above {PAIR_PROB_MIN:g}: {ens_empty:.4f}")
    print(f"  mean P(paired) per position, ensemble:        {kept.mean():.4f}")
    print(f"  mean top-{PAIR_TOP_K} mass vs 1 - P(unpaired):          "
          f"{kept.sum():.1f} / {(1.0 - pair_unpaired).sum():.1f}")


if __name__ == "__main__":
    sys.exit(main())
