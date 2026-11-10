#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Shared paths and settings for the tiled-amplicon benchmark.

Five arms -- Olivia-nondeg, Olivia-deg, Olivar 1.3.3, PrimalScheme3 and
varVAMP -- are run on the same alignment, at the same amplicon size and number
of pools. rerun_all.py drives all twelve targets through run_one.py, which
overrides MSA_PATH and OUT_DIR below per target, so nothing here needs editing
to switch targets.
'''

import os

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BENCH_DIR)

# Directory holding the `olivia` package. OLIVIA_SRC selects another package
# tree, e.g. to A/B a modified copy against this one.
SRC_DIR = os.environ.get('OLIVIA_SRC', os.path.join(REPO_DIR, 'src'))
PACKAGE_DIR = os.path.join(SRC_DIR, 'olivia')

# Per-target alignments, rebuilt from the accession lists under targets/
# (see benchmark/README.md).
TARGETS_DIR = os.path.join(BENCH_DIR, 'targets')

# Input and output of a single-target run (overridden by run_one.py)

MSA_PATH = os.path.join(TARGETS_DIR, 'hev', 'hev_stratified_aligned.fasta')
OUT_DIR = os.path.join(BENCH_DIR, 'output', 'HEV_400amp_newer')

PRIMER_TABLE = os.path.join(OUT_DIR, 'designed_primers.csv')   # distinct primers
POOL_TABLE = os.path.join(OUT_DIR, 'designed_pools.csv')       # pools as designed

# Design settings, common to every tool

AMPLICON_SIZES = (400,)
N_POOLS = 2
THREADS = 32
SEED = 10

# PrimalScheme3 --min-base-freq. At the 0.0 default PS3 requires a primer to
# match every sequence in the MSA, and several alignments yield no valid kmers.
PS3_MIN_BASE_FREQ = 0.05

# varVAMP -t and -a for a single-target run. rerun_all.py replaces both per
# target with the values chosen from the parameter sweep (paper Supplementary 6).
VARVAMP_THRESHOLD = 0.0   # 0.0 lets varVAMP estimate the threshold itself
VARVAMP_N_AMBIG = 30

# Olivia

ENCODER_MODEL = os.path.join(REPO_DIR, 'results', 'encoder', 'encoder.pt')

# Fraction of primer candidates the encoder filter keeps per plex and direction.
ENCODER_KEEP_FRAC = 0.5

# Hard per-candidate limits applied while generating primers (primer_gate.py).
OLIVIA_PRIMER_GATE = 'lenient'

# Maximum expansions of a degenerate primer candidate (varVAMP's -a).
OLIVIA_MAX_DEG = 128

# Extra tiling() settings for both Olivia arms.
if os.environ.get('OLIVIA_CURRENT') == '1':
    OLIVIA_TILING_EXTRA = {}
else:
    OLIVIA_TILING_EXTRA = {
        'canonical_folding': False,
        'fix_intraplex': False,
        'deg_sample': 0,
    }
