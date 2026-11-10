#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Collect every tool's primers into designed_primers.csv and designed_pools.csv.

    python collect_primers.py --out output/HEV_deg
'''

import argparse
import os
import sys

import pandas as pd

import config


def read_olivar(path):
    '''Olivar design CSV: one row per amplicon, forward and reverse in columns.'''
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        pool = int(r['pool'])
        rows.append((str(r['amplicon_id']) + '_LEFT', 'forward', r['fP'], pool))
        rows.append((str(r['amplicon_id']) + '_RIGHT', 'reverse', r['rP'], pool))
    return rows


def read_bed(path):
    '''PrimalScheme3 / varVAMP 7-column BED; all alternates are kept.'''
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            f = line.rstrip('\n').split('\t')
            if len(f) < 7:
                continue
            name, pool, strand, seq = f[3], f[4], f[5], f[6]
            rows.append((name, 'forward' if strand == '+' else 'reverse', seq, int(pool)))
    return rows


# tool -> (subdirectory under output/, file within the per-size dir, reader)
SOURCES = {
    'Olivia':        ('Olivia',        'design.csv',  read_olivar),
    'Olivar':        ('Olivar',        'design.csv',  read_olivar),
    'PrimalScheme3': ('PrimalScheme3', 'primer.bed',  read_bed),
    'varVAMP':       ('varVAMP',       'primers.bed', read_bed),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=None,
                    help='directory to collect [config.OUT_DIR]')
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out) if args.out else config.OUT_DIR
    primer_table = os.path.join(out_dir, 'designed_primers.csv')
    pool_table = os.path.join(out_dir, 'designed_pools.csv')

    records = []
    for tool, (subdir, fname, reader) in SOURCES.items():
        for amp_size in config.AMPLICON_SIZES:
            path = os.path.join(out_dir, subdir, 'amp%d' % amp_size, fname)
            if not os.path.isfile(path):
                print('  MISSING %s -- did %s run for amplicon %d?' % (path, tool, amp_size))
                continue
            for name, direction, seq, pool in reader(path):
                records.append({'tool': tool, 'amplicon_size': amp_size,
                                'pool': pool, 'name': name, 'direction': direction,
                                'primer': str(seq).upper().strip()})

    df = pd.DataFrame(records)
    if df.empty:
        raise SystemExit('no primers collected -- run the three design scripts first')

    # The three amplicon sizes tile the same gene, so a tool often picks the same primer
    # more than once.
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(pool_table, index=False)

    n_raw = len(df)
    df = df.drop_duplicates(subset=['tool', 'primer']).reset_index(drop=True)
    df.to_csv(primer_table, index=False)

    print('\n%d pool memberships -> %s' % (n_raw, pool_table))
    print('%d distinct primers (de-duplicated within tool) -> %s'
          % (len(df), primer_table))
    print()
    summary = df.groupby('tool').agg(
        n_primers=('primer', 'size'),
        mean_len=('primer', lambda s: round(s.str.len().mean(), 1)))
    print(summary.to_string())


if __name__ == '__main__':
    main()
