#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''The twelve viral targets of the benchmark: alignment, output directories and
degenerate-consensus threshold of each.
'''

import os

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
TARGETS_DIR = os.path.join(BENCH_DIR, 'targets')


class Target(object):
    def __init__(self, name, out_dir, msa_path, deg_dir=None, cons_thres=0.7):
        self.name = name
        self.out_dir = os.path.join(BENCH_DIR, 'output', out_dir)
        self.msa_path = msa_path
        self.deg_dir = deg_dir
        self.cons_thres = cons_thres

    @property
    def primer_table(self):
        return os.path.join(self.out_dir, 'designed_primers.csv')

    def __repr__(self):
        return 'Target(%s)' % self.name


def _msa(stem, folder=None):
    folder = folder or stem
    return os.path.join(TARGETS_DIR, folder, '%s_stratified_aligned.fasta' % stem)


# Ordered from the smallest alignment (genomes x columns) to the largest, so a
# run interrupted partway has finished as many targets as possible.
TARGETS = [
    Target('ratHEV',     'RATHEV_400amp',       _msa('rathev'),  'ratHEV_deg',       0.70),
    Target('HEV',        'HEV_400amp_newer',    _msa('hev'),     'HEV_deg',          0.70),
    Target('BDV-1',      'BDV1_400amp',         _msa('bdv1'),    'BDV1_deg_t0.9',    0.90),
    Target('HCV',        'HCV_400amp',          _msa('hcv'),     'HCV_deg',          0.70),
    Target('HAV',        'HAV_400amp',          _msa('hav'),     'HAV_deg_t0.75',    0.75),
    Target('Polio-1',    'POLIO1_400amp',       _msa('polio'),   'POLIO1_deg_t0.75', 0.75),
    Target('Rabies',     'RABIES_400amp',       _msa('rabies'),  'RABIES_deg',       0.70),
    Target('HIV-1',      'HIV1_400amp_newer',   _msa('hiv1'),    'HIV1_deg',         0.70),
    Target('Measles',    'MEASLES_400amp',      _msa('measles'), 'MEASLES_deg',      0.70),
    Target('RSV-A',      'RSVA_400amp',         _msa('rsva', 'rsva_gisaid'), 'RSVA_deg', 0.70),
    Target('RSV-B',      'RSVB_400amp',         _msa('rsvb', 'rsvb_gisaid'), 'RSVB_deg', 0.70),
    Target('SARS-CoV-2', 'SARS2_400amp_gisaid', _msa('sars2'),   'SARS2_deg',        0.70),
]

BY_NAME = dict((t.name, t) for t in TARGETS)


def select(names):
    '''names is a list of target names, or empty/None for all of them.'''
    if not names:
        return list(TARGETS)
    out = []
    for n in names:
        if n not in BY_NAME:
            raise SystemExit('unknown target %r; known: %s'
                             % (n, ', '.join(t.name for t in TARGETS)))
        out.append(BY_NAME[n])
    return out
