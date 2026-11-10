#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Run mismatch_analysis.py one target at a time and combine the results.

    python run_mismatch_all.py --out output/mismatch_deg_final
'''

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mismatch_analysis as ma

# Entropy order, most conserved first, minus HCV. Mirrors
# mismatch_analysis.TARGET_ORDER so the combined figure keeps that column order.
DEFAULT = [t for t in ma.TARGET_ORDER if t != 'HCV']

# The three tables a finished per-target run leaves behind.
PARTS = ['mismatch_pairs.csv', 'fig4b_mismatch_by_position.csv', 'summary.csv']


def slug(name):
    return name.replace(' ', '_').replace('/', '_')


def run_one(target, work, threads, extra):
    d = os.path.join(work, slug(target))
    if all(os.path.isfile(os.path.join(d, p)) for p in PARTS):
        print('  %-12s already done, skipping' % target)
        return d, True
    os.makedirs(d, exist_ok=True)
    cmd = [sys.executable, '-u', os.path.join(HERE, 'mismatch_analysis.py'),
           '--targets', target, '--threads', str(threads), '--out', d] + extra
    t0 = time.time()
    with open(os.path.join(d, 'run.log'), 'wb') as fh:
        p = subprocess.run(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
    ok = p.returncode == 0 and all(
        os.path.isfile(os.path.join(d, x)) for x in PARTS)
    print('  %-12s %s  %.1f s%s' % (target, 'ok' if ok else 'FAILED',
                                    time.time() - t0,
                                    '' if ok else '  (see %s/run.log)' % d))
    return d, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--targets', nargs='*', default=None)
    ap.add_argument('--out', default=os.path.join(HERE, 'output', 'mismatch_deg_final'),
                    help='combined result directory (the one the paper cites)')
    ap.add_argument('--work', default=None,
                    help='per-target directory [<out>/per_target]')
    ap.add_argument('--threads', type=int, default=32)
    ap.add_argument('--force', action='store_true',
                    help='re-run targets whose output already exists')
    args, extra = ap.parse_known_args()

    targets = args.targets or DEFAULT
    out = os.path.abspath(args.out)
    work = os.path.abspath(args.work) if args.work else os.path.join(out, 'per_target')
    os.makedirs(work, exist_ok=True)

    if args.force:
        for t in targets:
            d = os.path.join(work, slug(t))
            for p in PARTS:
                try:
                    os.remove(os.path.join(d, p))
                except OSError:
                    pass

    print('%d target(s), %d threads each -> %s\n' % (len(targets), args.threads, work))
    dirs, failed = [], []
    for t in targets:
        d, ok = run_one(t, work, args.threads, extra)
        (dirs if ok else failed).append(d if ok else t)

    if not dirs:
        sys.exit('no target produced output')
    if failed:
        print('\nFAILED: %s' % ', '.join(failed))

    # Concatenate, preserving each part's own header once.
    import pandas as pd
    print('\ncombining %d target(s) -> %s' % (len(dirs), out))
    os.makedirs(out, exist_ok=True)
    for part in PARTS:
        frames = [pd.read_csv(os.path.join(d, part)) for d in dirs]
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(os.path.join(out, part), index=False)
        print('  %-34s %7d rows' % (part, len(df)))

    # Redraw from the combined tables with the module's own renderer.
    cmd = [sys.executable, '-u', os.path.join(HERE, 'mismatch_analysis.py'),
           '--replot-from', out, '--out', out] + extra
    p = subprocess.run(cmd, cwd=HERE)
    if p.returncode != 0:
        sys.exit('replot failed')
    print('\ndone -> %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
