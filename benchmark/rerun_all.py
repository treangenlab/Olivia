#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Run the full benchmark: every tool on every target in bench_targets.py.

    python rerun_all.py --list
    python rerun_all.py --stage all
    python rerun_all.py --stage designs --targets HEV HCV

varVAMP's -t/-a per target come from the sweep: the lowest -a reaching 90%
coverage, then the highest -t.
'''

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time

import bench_targets

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, 'output', 'varvamp_sweep_2026-10-31', 'sweep_results.csv')
LOGDIR = os.path.join(HERE, 'output', 'logs')

THREADS = 32
PS3_MIN_BASE_FREQ = 0.05
MIN_COVERAGE = 90.0

# Conda environment of each tool (environment files in envs/).
OLIVIA_ENV = 'olivia'
TOOL_ENV = [('Olivia', OLIVIA_ENV), ('Olivar', 'olivar'),
            ('PrimalScheme3', 'primalscheme3'), ('varVAMP', 'varvamp')]

# (name, sweep label, aligned MSA, non-deg out dir, deg out dir, deg cons-thres)
TARGETS = [(t.name, t.name, t.msa_path, os.path.basename(t.out_dir), t.deg_dir,
            t.cons_thres) for t in bench_targets.TARGETS]


def _b(*p):
    return os.path.join(HERE, *p)


def varvamp_settings():
    '''label -> (t, a, coverage, met_floor), by the rule in the docstring.'''
    if not os.path.exists(SWEEP):
        sys.exit('no sweep results at %s -- run varvamp_param_sweep.py first' % SWEEP)
    rows = {}
    for r in csv.DictReader(open(SWEEP)):
        try:
            cov = float(r['pct_coverage'])
        except (TypeError, ValueError):
            continue
        t = r['t'].strip()
        rows.setdefault(r['target'], []).append(
            (t if t else 'auto', int(r['a']), cov))
    out = {}
    for tgt, cands in rows.items():
        # -t 'auto' sorts as the highest threshold: varVAMP estimates it per
        # alignment and it lands near the top of the grid (0.91-0.97 observed).
        key = lambda c: (c[1], -(99.0 if c[0] == 'auto' else float(c[0])))
        ok = [c for c in cands if c[2] >= MIN_COVERAGE]
        if ok:
            pick = sorted(ok, key=key)[0]
            out[tgt] = (pick[0], pick[1], pick[2], True)
        else:
            best = max(c[2] for c in cands)
            pick = sorted([c for c in cands if c[2] == best], key=key)[0]
            out[tgt] = (pick[0], pick[1], pick[2], False)
    return out


def run(cmd, log, label):
    '''Run cmd, append stdout+stderr to log, return (ok, seconds).'''
    os.makedirs(os.path.dirname(log), exist_ok=True)
    t0 = time.time()
    with open(log, 'ab') as fh:
        fh.write(('\n===== %s :: %s\n' % (time.strftime('%F %T'), label)).encode())
        fh.flush()
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=HERE,
                           env=_single_threaded_blas_env())
    return p.returncode == 0, time.time() - t0


def _single_threaded_blas_env():
    '''os.environ with the BLAS/OpenMP runtimes pinned to one thread.'''
    env = dict(os.environ)
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[var] = '1'
    return env


# The file a finished run of each tool leaves behind. Presence of this file is
# what "done" means; anything less is a run that died partway.
DONE_FILE = {
    'Olivia':        'amp400/design.primer.bed',
    'Olivar':        'amp400/design.primer.bed',
    'PrimalScheme3': 'amp400/primer.bed',
    'varVAMP':       'amp400/primers.tsv',
}


def done_marker(path, msa, tool=None):
    '''True only if `path` holds a complete run newer than its alignment.'''
    if not (os.path.isdir(path) and os.listdir(path)):
        return False
    try:
        if os.path.getmtime(path) < os.path.getmtime(msa):
            return False
    except OSError:
        return False
    rel = DONE_FILE.get(tool) if tool else None
    if rel is None:
        return True
    return os.path.isfile(os.path.join(path, rel))


def stage_designs(targets, vv, force):
    for name, label, msa, out, _deg, _dt in targets:
        if label not in vv:
            print('  SKIP %s -- no sweep row (sweep it first)' % name)
            continue
        t, a, cov, met = vv[label]
        out_dir = _b('output', out)
        print('\n## %s  (varVAMP -t %s -a %d, %.1f%% coverage%s)'
              % (name, t, a, cov, '' if met else '  [BELOW FLOOR]'))
        for tool, env in TOOL_ENV:
            tool_dir = os.path.join(out_dir, tool)
            if not force and done_marker(tool_dir, msa, tool):
                print('    %-14s already present, skipping' % tool)
                continue
            # A partial directory from a killed run would otherwise be resumed
            # into; Olivar in particular reuses a half-built .olvr reference.
            if os.path.isdir(tool_dir):
                shutil.rmtree(tool_dir)
                print('    %-14s clearing incomplete run' % tool)
            cmd = ['conda', 'run', '--no-capture-output', '-n', env,
                   'python', 'run_one.py', tool, msa, out_dir,
                   str(THREADS), str(PS3_MIN_BASE_FREQ), str(t), str(a)]
            ok, secs = run(cmd, os.path.join(LOGDIR, 'designs', '%s.log' % name),
                           '%s / %s' % (name, tool))
            print('    %-14s %s  %.1f s' % (tool, 'ok' if ok else 'FAILED', secs))


def stage_deg(targets, force):
    for name, _label, msa, out, deg, dt in targets:
        deg_dir = _b('output', deg)
        if not force and done_marker(os.path.join(deg_dir, 'Olivia'), msa, 'Olivia'):
            print('  %-12s deg already present, skipping' % name)
            continue
        # Run through `python -c` so forkserver children do not re-run the design.
        code = ('import sys, config; config.THREADS = %d; '
                'sys.argv = ["design_olivia_deg.py", "--msa", %r, "--out", %r, '
                '"--cons-thres", %r, "--carry-from", %r]; '
                'import design_olivia_deg as m; m.main()'
                % (THREADS, msa, deg_dir, str(dt), _b('output', out)))
        cmd = ['conda', 'run', '--no-capture-output', '-n', OLIVIA_ENV,
               'python', '-c', code]
        ok, secs = run(cmd, os.path.join(LOGDIR, 'deg', '%s.log' % name),
                       '%s deg cons-thres=%s threads=%d' % (name, dt, THREADS))
        print('  %-12s deg t=%.2f  %s  %.1f s'
              % (name, dt, 'ok' if ok else 'FAILED', secs))


def stage_collect(targets):
    for name, _label, _msa, out, deg, _dt in targets:
        for out_dir in (_b('output', out), _b('output', deg)):
            ok, secs = run(['conda', 'run', '--no-capture-output', '-n', OLIVIA_ENV,
                            'python', 'collect_primers.py', '--out', out_dir],
                           os.path.join(LOGDIR, 'collect', '%s.log' % name),
                           '%s collect %s' % (name, os.path.basename(out_dir)))
            print('  %-12s collect %-20s %s  %.1f s'
                  % (name, os.path.basename(out_dir), 'ok' if ok else 'FAILED', secs))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', default='all',
                    choices=['all', 'designs', 'deg', 'collect'])
    ap.add_argument('--targets', nargs='*', default=None)
    ap.add_argument('--force', action='store_true',
                    help='rerun steps whose output already exists')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    vv = varvamp_settings()
    sel = TARGETS
    if args.targets:
        want = set(args.targets)
        sel = [t for t in TARGETS if t[0] in want]
        missing = want - {t[0] for t in TARGETS}
        if missing:
            sys.exit('unknown targets: %s' % ', '.join(sorted(missing)))

    if args.list:
        print('%-12s %-8s %-4s %8s %6s  %s'
              % ('target', '-t', '-a', 'coverage', 'floor', 'out dir'))
        for name, label, _msa, out, deg, dt in sel:
            if label in vv:
                t, a, cov, met = vv[label]
                print('%-12s %-8s %-4d %7.1f%% %6s  %s | %s t=%.2f'
                      % (name, t, a, cov, 'ok' if met else 'BELOW', out, deg, dt))
            else:
                print('%-12s %-8s %-4s %8s %6s  %s' % (name, '?', '?', '-', '-', out))
        return 0

    os.makedirs(LOGDIR, exist_ok=True)
    if args.stage in ('all', 'designs'):
        print('\n########## non-deg designs')
        stage_designs(sel, vv, args.force)
    if args.stage in ('all', 'deg'):
        print('\n########## degenerate designs')
        stage_deg(sel, args.force)
    if args.stage in ('all', 'collect'):
        print('\n########## collect')
        stage_collect(sel)
    print('\nlogs under %s' % LOGDIR)
    return 0


if __name__ == '__main__':
    sys.exit(main())
