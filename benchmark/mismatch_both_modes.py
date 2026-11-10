#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Primer-vs-alignment mismatches with Olivia in both design modes (paper Figs. 3-4).

    python mismatch_both_modes.py --threads 16
'''

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import deg_abnormality as da
import mismatch_analysis as ma

# The non-degenerate run directory per target.
NONDEG_DIRS = da.NONDEG_DIRS

EXTERNAL = ('Olivar', 'PrimalScheme3', 'varVAMP')

# Olivia's two modes first, then the externals. mismatch_analysis._render puts
# arms it does not know after the ones it does, so this replaces its ARM_ORDER.
ARM_ORDER = ['Olivia-deg', 'Olivia-nondeg', 'Olivar', 'PrimalScheme3', 'varVAMP']


def build_target_table(target, deg_dir, nondeg_dir, out_root, work_dir):
    '''Primer table with both Olivia modes for one target -> (path, notes).'''
    notes = []
    deg_path = os.path.join(out_root, deg_dir, 'designed_primers.csv') if deg_dir else None
    nd_path = os.path.join(out_root, nondeg_dir, 'designed_primers.csv') if nondeg_dir else None
    if not deg_path or not os.path.isfile(deg_path):
        return None, ['no deg run (%s)' % deg_path]
    if not nd_path or not os.path.isfile(nd_path):
        return None, ['no non-deg run (%s)' % nd_path]

    deg = pd.read_csv(deg_path)
    nd = pd.read_csv(nd_path)

    for tool in EXTERNAL:
        a = set(deg.loc[deg['tool'] == tool, 'primer'])
        b = set(nd.loc[nd['tool'] == tool, 'primer'])
        if a and b and a != b:
            notes.append('%s differs between the two runs; the non-deg copy is used' % tool)

    parts = []
    for src, tool, label in ((deg, 'Olivia', 'Olivia-deg'),
                             (nd, 'Olivia', 'Olivia-nondeg')):
        sub = src[src['tool'] == tool].copy()
        if sub.empty:
            notes.append('no %s arm' % label)
            continue
        sub['tool'] = label
        parts.append(sub)
    ext = nd[nd['tool'].isin(EXTERNAL)].copy()
    if not ext.empty:
        parts.append(ext)

    if not parts:
        return None, notes
    table = pd.concat(parts, ignore_index=True)
    path = os.path.join(work_dir, '%s_primers.csv'
                        % target.replace(' ', '_').replace('-', ''))
    table.to_csv(path, index=False)
    return path, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(HERE, 'output', 'mismatch_both_modes'))
    ap.add_argument('--targets', nargs='*', default=None)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--ncols', type=int, default=3)
    ap.add_argument('--ncols-4b', type=int, default=None)
    ap.add_argument('--min-score-frac', type=float, default=0.6)
    ap.add_argument('--min-support', type=float, default=0.2)
    ap.add_argument('--ambiguous', choices=ma.AMBIGUOUS_MODES, default='resolve')
    ap.add_argument('--arms', nargs='*', default=None)
    ap.add_argument('--replot-from', default=None)
    ap.add_argument('--compare-with', default=os.path.join(
        HERE, 'output', 'mismatch_deg_final', 'summary.csv'),
        help='published deg summary.csv, read only, for the agreement check')
    args = ap.parse_args()

    work_dir = os.path.join(args.out, 'primer_tables')
    os.makedirs(work_dir, exist_ok=True)

    # Both Olivia modes are new arm names, so the module's panel order has to
    # know about them. Set on the module rather than edited into it.
    ma.ARM_ORDER = ARM_ORDER

    if args.replot_from:
        pairs = pd.read_csv(os.path.join(args.replot_from, 'mismatch_pairs.csv'))
        positions = pd.read_csv(os.path.join(
            args.replot_from, 'fig4b_mismatch_by_position.csv'))
        summary = pd.read_csv(os.path.join(args.replot_from, 'summary.csv'))
        ma._render(pairs, positions, summary, args)
        return

    target_order = [t for t in ma.TARGET_ORDER
                    if not args.targets or t in args.targets]
    unknown = [t for t in (args.targets or []) if t not in ma.TARGET_ORDER]
    if unknown:
        raise SystemExit('unknown target(s): %s; known: %s'
                         % (', '.join(unknown), ', '.join(ma.TARGET_ORDER)))

    out_root = os.path.join(HERE, 'output')
    print('run directories per target')
    jobs = []
    for target in target_order:
        msa_path, deg_dir = ma.TARGETS[target]
        deg_dir = os.path.basename(deg_dir.rstrip('/'))
        nd_dir = NONDEG_DIRS.get(target)
        path, notes = build_target_table(target, deg_dir, nd_dir, out_root, work_dir)
        print('  %-11s deg=%-18s non-deg=%-22s %s'
              % (target, deg_dir, nd_dir, '; '.join(notes) if notes else ''))
        if path:
            jobs.append((target, msa_path, [('', path)]))

    if not jobs:
        raise SystemExit('no target has both runs')

    all_pairs, all_positions, summary = [], [], []
    for target, msa_path, specs in jobs:
        p, q, s = ma.run_one_target(target, msa_path, specs, args)
        all_pairs += p
        all_positions += q
        summary += s

    if not all_pairs:
        raise SystemExit('nothing could be placed')

    ma._render(pd.concat(all_pairs, ignore_index=True),
               pd.concat(all_positions, ignore_index=True),
               pd.DataFrame(summary), args)

    # Check Olivia-deg against the published deg summary.
    if os.path.isfile(args.compare_with):
        pub = pd.read_csv(args.compare_with)
        new = pd.read_csv(os.path.join(args.out, 'summary.csv'))
        rows = []
        for _, r in new[new['arm'] == 'Olivia-deg'].iterrows():
            m = pub[(pub['virus'] == r['virus']) & (pub['arm'] == 'Olivia')]
            if len(m):
                rows.append({'virus': r['virus'],
                             'published': float(m['mean_mismatches_per_primer'].iloc[0]),
                             'here': float(r['mean_mismatches_per_primer'])})
        if rows:
            chk = pd.DataFrame(rows)
            chk['delta'] = chk['here'] - chk['published']
            print('\n--- Olivia-deg vs the published deg summary (mean mismatches) ---')
            print(chk.round(4).to_string(index=False))
            worst = chk['delta'].abs().max()
            print('largest absolute difference: %.4f%s' % (
                worst, '  -- expected ~0' if worst < 1e-6 else
                '  -- NOT zero, check which run each side scored'))

    print('\nwrote %s' % args.out)


if __name__ == '__main__':
    main()
