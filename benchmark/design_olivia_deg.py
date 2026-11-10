#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Design tiled schemes with Olivia in degenerate mode (Olivia-deg).

    python design_olivia_deg.py --targets HEV HCV
    python design_olivia_deg.py --msa aln.fasta --out output/custom_deg --cons-thres
    0.75
'''

import argparse
import os
import shutil

import config
import bench_targets
from design_olivia import build_reference, design


def run_target(name, msa, out, amp_size, cons_thres=0.7, competitors_from=None,
               extra=None):
    '''One target: build the degenerate reference once, then design.'''
    if not os.path.isfile(msa):
        raise SystemExit('missing alignment for %s: %s' % (name, msa))
    os.makedirs(out, exist_ok=True)

    if competitors_from and os.path.isdir(competitors_from):
        for tool in ('Olivar', 'PrimalScheme3', 'varVAMP'):
            src = os.path.join(competitors_from, tool)
            dst = os.path.join(out, tool)
            if os.path.isdir(src) and not os.path.isdir(dst):
                print('--- carrying %s over from %s' % (tool, competitors_from))
                shutil.copytree(src, dst)

    # cons_thres changes the reference, so it goes in the directory name
    work_dir = os.path.join(out, 'work_deg' if cons_thres == 0.7
                            else 'work_deg_t%g' % cons_thres)
    print('\n=== %s: building reference (deg=True, cons_thres=%g) ===' % (name, cons_thres))
    ref_path = build_reference(work_dir, msa, deg=True, cons_thres=cons_thres)

    out_dir = os.path.join(out, 'Olivia', 'amp%d' % amp_size)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    print('\n=== %s: Olivia-deg (encoder_keep_frac=%.2f), amplicon %d ==='
          % (name, config.ENCODER_KEEP_FRAC, amp_size))
    design(ref_path, out_dir, 'design', amp_size, config.ENCODER_KEEP_FRAC,
           deg=True, extra=extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--targets', nargs='*', default=None,
                    help='targets from bench_targets.py [all twelve]')
    ap.add_argument('--msa', default=None,
                    help='one-off alignment, instead of --targets')
    ap.add_argument('--out', default=None, help='output directory for --msa')
    ap.add_argument('--amp-size', type=int, default=400, help='amplicon size [400]')
    ap.add_argument('--cons-thres', type=float, default=None, metavar='T',
                    help='degenerate-consensus threshold [per target from '
                         'bench_targets.py; 0.7 for --msa]')
    ap.add_argument('--no-carry', action='store_true',
                    help='do not copy the competitors from the non-degenerate run')
    ap.add_argument('--carry-from', default=None, metavar='DIR',
                    help='with --msa: non-degenerate output directory to copy '
                         'the competitors from')
    ap.add_argument('--deg-sample', type=int, default=None, metavar='K',
                    help='fold at most K expansion pairs per candidate pair '
                         '[config.OLIVIA_TILING_EXTRA, 0 = exact]')
    ap.add_argument('--no-precompute', action='store_true',
                    help='fill the primer-pair table lazily during annealing')
    ap.add_argument('--precompute-batch', type=int, default=None, metavar='N',
                    help='concrete pairs per precompute batch [2000000]; '
                         'lower it to reduce peak memory')
    args = ap.parse_args()

    extra = {}
    if args.deg_sample is not None:
        extra['deg_sample'] = args.deg_sample
    if args.no_precompute:
        extra['precompute_pairs'] = False
    if args.precompute_batch is not None:
        extra['precompute_batch'] = args.precompute_batch

    if args.msa:
        out = args.out or os.path.join(config.BENCH_DIR, 'output', 'deg_oneoff')
        run_target('custom', args.msa, out, args.amp_size,
                   cons_thres=0.7 if args.cons_thres is None else args.cons_thres,
                   competitors_from=None if args.no_carry else args.carry_from,
                   extra=extra)
        print('\ndone. design under %s' % out)
        return

    for t in bench_targets.select(args.targets):
        cons_thres = t.cons_thres if args.cons_thres is None else args.cons_thres
        out = os.path.join(config.BENCH_DIR, 'output', t.deg_dir)
        run_target(t.name, t.msa_path, out, args.amp_size, cons_thres=cons_thres,
                   competitors_from=None if args.no_carry else t.out_dir,
                   extra=extra)
        print('\n### %s finished -> %s' % (t.name, out))
        print('    next: python collect_primers.py --out %s' % out)


if __name__ == '__main__':
    main()
