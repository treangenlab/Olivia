#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Sweep varVAMP's -t / -a on an alignment and report the setting to keep.

    python varvamp_param_sweep.py --msa RSV-A=/path/to/aligned.fasta
'''

import argparse
import collections
import csv
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# (-t, -a) grid; t=None omits -t so varVAMP estimates it.
GRID = [(None, 1), (0.90, 1), (0.80, 4), (0.70, 4), (0.70, 6), (0.60, 8), (0.40, 10)]

SETS = {
    'rsv': [
        ('RSV-A', os.path.join(HERE, 'rsva_gisaid', 'rsva_gisaid_aligned.fasta')),
        ('RSV-B', os.path.join(HERE, 'rsvb_gisaid', 'rsvb_gisaid_aligned.fasta')),
    ],
    'flu-h1n1': [
        ('Flu-H1N1-seg%d-%s' % (seg, gene),
         os.path.join(HERE, 'flu_gisaid', 'h1n1', 'h1n1_seg%d_%s_aligned.fasta' % (seg, gene)))
        for seg, gene in [(1, 'PB2'), (2, 'PB1'), (3, 'PA'), (4, 'HA'),
                          (5, 'NP'), (6, 'NA'), (7, 'MP'), (8, 'NS')]
    ],
    'flu-h3n2': [
        ('Flu-H3N2-seg%d-%s' % (seg, gene),
         os.path.join(HERE, 'flu_gisaid', 'h3n2', 'h3n2_seg%d_%s_aligned.fasta' % (seg, gene)))
        for seg, gene in [(1, 'PB2'), (2, 'PB1'), (3, 'PA'), (4, 'HA'),
                          (5, 'NP'), (6, 'NA'), (7, 'MP'), (8, 'NS')]
    ],
}

AMBIG = set('RYKMSWBVDHN')


def read_consensus(path):
    seq = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith('>'):
                seq.append(line.strip())
    return ''.join(seq).upper()


def union_length(spans):
    total, end = 0, -1
    for a, b in sorted(spans):
        if b <= end:
            continue
        total += b - max(a, end)
        end = max(end, b)
    return total


def measure(out_dir):
    '''(n_amplicons, pct_coverage, pct_ambiguous, consensus_len) from a run.'''
    bed = os.path.join(out_dir, 'amplicons.bed')
    cons = os.path.join(out_dir, 'ambiguous_consensus.fasta')
    if not os.path.isfile(bed) or not os.path.isfile(cons):
        return None
    spans = []
    with open(bed) as fh:
        for line in fh:
            f = line.rstrip('\n').split('\t')
            if len(f) >= 3 and f[1].isdigit():
                spans.append((int(f[1]), int(f[2])))
    seq = read_consensus(cons)
    if not seq:
        return None
    n_amb = sum(c in AMBIG for c in seq)
    return (len(spans), 100.0 * union_length(spans) / len(seq),
            100.0 * n_amb / len(seq), len(seq))


def logged_threshold(out_dir):
    '''The threshold varVAMP actually used, from the "THRESHOLD = x" it logs.'''
    log = os.path.join(out_dir, 'varvamp_log.txt')
    if not os.path.isfile(log):
        return None
    for line in open(log, errors='replace'):
        m = re.match(r'\s*THRESHOLD\s*=\s*([0-9.]+)', line)
        if m:
            return float(m.group(1))
    return None


def run_one(label, msa, t, a, args):
    tag = 'auto' if t is None else ('%.2f' % t).rstrip('0').rstrip('.')
    out_dir = os.path.join(args.out, label, 't%s_a%d' % (tag, a))
    if os.path.isdir(out_dir):
        if args.reuse and measure(out_dir):
            print('    t=%-4s a=%-2d  reusing existing run' % (tag, a))
            return out_dir, 0.0, True
        shutil.rmtree(out_dir)
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)

    cmd = ['conda', 'run', '--no-capture-output', '-n', args.env,
           'varvamp', 'tiled',
           '--threads', str(args.threads),
           '--opt-length', str(args.amp_size),
           '--max-length', str(int(round(args.amp_size * 1.1))),
           '--overlap', str(args.overlap),
           '-a', str(a)]
    # Omitting -t entirely is what makes varVAMP estimate it; see GRID.
    if t is not None:
        cmd += ['-t', str(t)]
    cmd += [msa, out_dir]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or '').strip().splitlines()[-3:]
        print('    t=%-4s a=%-2d  FAILED in %.0fs: %s'
              % (tag, a, dt, ' | '.join(tail)[:200]))
        return None, dt, False
    return out_dir, dt, False


def pick(rows):
    '''Most coverage, then most amplicons, then the most conservative settings.'''
    if not rows:
        return None
    def key(r):
        t = r['t'] if r['t'] is not None else (r['logged_threshold'] or 0.0)
        return (round(r['pct_coverage'], 1), r['n_amplicons'], t, -r['a'])
    return max(rows, key=key)


def label_of(row):
    '''How a setting is written in the manuscript's parameter table.'''
    if row['t'] is None:
        return ('auto (%.2f)' % row['logged_threshold']
                if row['logged_threshold'] is not None else 'auto')
    return '%.2f' % row['t']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', dest='sets', nargs='*', choices=sorted(SETS),
                    help='named target set(s) to sweep')
    ap.add_argument('--msa', action='append', default=None, metavar='LABEL=PATH',
                    help='an ad-hoc alignment instead of --set')
    ap.add_argument('--out', default=os.path.join(HERE, 'output', 'varvamp_sweep'))
    ap.add_argument('--amp-size', type=int, default=400)
    ap.add_argument('--overlap', type=int, default=25)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--env', default='varvamp')
    ap.add_argument('--grid', default=None,
                    help='override, e.g. "0.0/1,0.7/4,0.7/6"')
    ap.add_argument('--reuse', action='store_true',
                    help='keep runs already on disk instead of redoing them')
    args = ap.parse_args()

    targets = []
    for name in (args.sets or []):
        targets += SETS[name]
    for spec in (args.msa or []):
        label, _, path = spec.partition('=')
        targets.append((label or os.path.basename(path).split('.')[0], path))
    if not targets:
        raise SystemExit('nothing to sweep: pass --set and/or --msa')

    grid = GRID
    if args.grid:
        grid = []
        for part in args.grid.split(','):
            t, _, a = part.partition('/')
            grid.append((float(t), int(a)))

    missing = [(l, p) for l, p in targets if not os.path.isfile(p)]
    if missing:
        raise SystemExit('missing alignment(s):\n  %s'
                         % '\n  '.join('%s -> %s' % m for m in missing))

    os.makedirs(args.out, exist_ok=True)
    print('sweeping %d setting(s) on %d alignment(s), amplicon %d nt, %d threads\n'
          % (len(grid), len(targets), args.amp_size, args.threads))

    all_rows = []
    for label, msa in targets:
        print('### %s\n    %s' % (label, msa))
        rows = []
        for t, a in grid:
            out_dir, dt, reused = run_one(label, msa, t, a, args)
            if not out_dir:
                continue
            m = measure(out_dir)
            if not m:
                print('    t=%-4s a=%-2d  produced no amplicons' % (t, a))
                continue
            n_amp, cov, amb, clen = m
            row = {'target': label, 't': t, 'a': a,
                   'logged_threshold': logged_threshold(out_dir),
                   'n_amplicons': n_amp, 'pct_coverage': cov,
                   'pct_ambiguous_consensus': amb, 'consensus_len': clen,
                   'seconds': round(dt, 1), 'out_dir': out_dir}
            rows.append(row)
            if not reused:
                print('    t=%-4s a=%-2d  %3d amplicons  %5.1f%% coverage  '
                      '%4.1f%% ambiguous  %5.0fs'
                      % ('auto' if t is None else t, a, n_amp, cov, amb, dt))
        best = pick(rows)
        for r in rows:
            r['chosen'] = (r is best)
        all_rows += rows
        if best:
            print('    -> keep -t %s -a %d  (%d amplicons, %.1f%% coverage)\n'
                  % (label_of(best), best['a'], best['n_amplicons'],
                     best['pct_coverage']))
        else:
            print('    -> no setting tiled this alignment\n')

    if not all_rows:
        raise SystemExit('no run produced amplicons')

    # Rows for targets this call did not sweep are carried over, so sweeping one
    # set does not discard another's results.
    path = os.path.join(args.out, 'sweep_results.csv')
    fields = list(all_rows[0].keys())
    kept = []
    if os.path.isfile(path):
        swept = {r['target'] for r in all_rows}
        with open(path, newline='') as fh:
            kept = [r for r in csv.DictReader(fh) if r.get('target') not in swept]
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in kept:
            w.writerow({k: r.get(k, '') for k in fields})
        w.writerows(all_rows)

    print('=== settings to keep ===')
    print('%-22s %-12s %-4s %-10s %-9s %s'
          % ('target', '-t', '-a', 'amplicons', 'coverage', 'ambiguous'))
    for label, _ in targets:
        rows = [r for r in all_rows if r['target'] == label]
        best = pick(rows)
        if not best:
            print('%-22s  no setting tiled this alignment' % label)
            continue
        print('%-22s %-12s %-4d %-10d %-9.1f %.1f%%'
              % (label, label_of(best), best['a'], best['n_amplicons'],
                 best['pct_coverage'], best['pct_ambiguous_consensus']))
    print('\nwrote %s' % path)


if __name__ == '__main__':
    main()
