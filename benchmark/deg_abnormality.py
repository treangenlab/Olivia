#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Abnormality rate of every arm at the published and degeneracy-matched thresholds.

    python deg_abnormality.py

Matched thresholds reweight the held-out calibration corpus to each arm's
degenerate-base histogram before taking the 95th percentile.
'''

import argparse
import collections
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config

sys.path.insert(0, config.SRC_DIR)
sys.path.insert(0, config.PACKAGE_DIR)

AMBIGUOUS = set('RYKMSWBVDHN')
HEADS = ('A', 'C_marg', 'C_inter', 'P')
HEAD_LABEL = {'A': 'Head A', 'C_marg': 'Head C (Marg.)',
              'C_inter': 'Head C (Inter.)', 'P': 'Head P'}

# The non-degenerate run directory for each target (bench_targets.py).
NONDEG_DIRS = {
    'SARS-CoV-2': 'SARS2_400amp_gisaid',
    'Measles':    'MEASLES_400amp',
    'BoDV-1':     'BDV1_400amp',
    'HAV':        'HAV_400amp',
    'Polio':      'POLIO1_400amp',
    'Rabies':     'RABIES_400amp',
    'rat HEV':    'RATHEV_400amp',
    'HIV-1':      'HIV1_400amp_newer',
    'HEV':        'HEV_400amp_newer',
    'HCV':        'HCV_400amp',
    'RSV-A':      'RSVA_400amp',
    'RSV-B':      'RSVB_400amp',
}

# Tools to report, in table order.
TOOLS = ('Olivia', 'Olivar', 'PrimalScheme3', 'varVAMP')


def n_degenerate(seq):
    return sum(c in AMBIGUOUS for c in str(seq).upper())


def canonical_deg_dirs():
    '''Degenerate run directory per target (as in mismatch_analysis.TARGETS).'''
    import mismatch_analysis as ma
    return {name: out_dir for name, (_msa, out_dir) in ma.TARGETS.items()}, ma.TARGET_ORDER


# Calibration pool

def load_calibration_pool(args, rng):
    '''Held-out real primers to recalibrate on: all degenerate ones, non-degenerate
    subsampled.
    '''
    seqs, ndeg = [], []
    for split in args.splits:
        path = os.path.join(args.calib_dir, '%s.csv' % split)
        if not os.path.isfile(path):
            raise SystemExit('no calibration split at %s' % path)
        n_rows = 0
        for chunk in pd.read_csv(path, usecols=['cleaned_primer'], chunksize=200000):
            for s in chunk['cleaned_primer']:
                if not isinstance(s, str) or not s.strip():
                    continue
                s = s.upper().strip()
                seqs.append(s)
                ndeg.append(n_degenerate(s))
                n_rows += 1
        print('  %-5s %d primers' % (split, n_rows))

    seqs = np.asarray(seqs, dtype=object)
    ndeg = np.asarray(ndeg)

    # The corpus's own stratum counts, before any subsampling.
    corpus_strata = collections.Counter(ndeg.tolist())

    keep = np.flatnonzero(ndeg > 0)
    zero = np.flatnonzero(ndeg == 0)
    if len(zero) > args.n_zero:
        zero = rng.choice(zero, args.n_zero, replace=False)
    idx = np.concatenate([keep, zero])
    print('calibration pool: %d degenerate + %d non-degenerate = %d primers'
          % (len(keep), len(zero), len(idx)))
    return list(seqs[idx]), ndeg[idx], corpus_strata


def score_pool(scorer, seqs, cache, label):
    '''Head scores for a list of primers, cached to .npz under the out dir.'''
    if cache and os.path.isfile(cache):
        z = np.load(cache, allow_pickle=True)
        if int(z['n']) == len(seqs):
            print('  %s: reusing %s' % (label, os.path.basename(cache)))
            return {h: z[h] for h in HEADS}
        print('  %s: cache is for %d primers, rescoring %d'
              % (label, int(z['n']), len(seqs)))
    print('  %s: scoring %d primers' % (label, len(seqs)))
    s = scorer.score(seqs)
    out = {h: np.asarray(s[h], dtype=np.float64) for h in HEADS}
    if cache:
        np.savez(cache, n=len(seqs), **out)
    return out


# Degeneracy-matched thresholds

def strata_of(ndeg, cap):
    '''Degenerate-base count, with everything at or above cap pooled into cap.'''
    return np.minimum(np.asarray(ndeg), cap)


def weighted_quantile(values, weights, q):
    '''The q-quantile of values under weights, by linear interpolation of the
    weighted CDF. Weights need not sum to one.'''
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float('nan')
    # Midpoint of each primer's weight interval, which is the convention that
    # makes the unweighted case agree with np.percentile's 'linear' method.
    p = (cw - 0.5 * w) / cw[-1]
    return float(np.interp(q, p, v))


def matched_thresholds(arm_ndeg, pool_ndeg, pool_scores, cap, quantile):
    '''Per-head thresholds matched to the arm's degeneracy, with support diagnostics.'''
    arm_count = collections.Counter(strata_of(arm_ndeg, cap).tolist())
    return thresholds_for_histogram(arm_count, pool_ndeg, pool_scores, cap, quantile)


def thresholds_for_histogram(arm_count, pool_ndeg, pool_scores, cap, quantile):
    '''Per-head quantiles over the pool, reweighted to the arm's degenerate-base histogram.
    '''
    pool_s = strata_of(pool_ndeg, cap)
    pool_count = collections.Counter(pool_s.tolist())

    capped = collections.Counter()
    for k, v in arm_count.items():
        capped[int(min(k, cap))] += v
    arm_count = capped
    n_arm = sum(arm_count.values())

    folded = {}
    share = collections.Counter()
    for s, k in arm_count.items():
        t = s
        while t > 0 and pool_count.get(t, 0) == 0:
            t -= 1
        if t != s:
            folded[int(s)] = int(t)
        share[t] += k

    w = np.zeros(len(pool_s), dtype=np.float64)
    for s, k in share.items():
        sel = pool_s == s
        w[sel] = (k / n_arm) / pool_count[s]

    th = {h: weighted_quantile(pool_scores[h], w, quantile) for h in HEADS}
    ess = float(w.sum() ** 2 / np.sum(w ** 2)) if np.any(w > 0) else 0.0
    support = {int(s): int(pool_count.get(s, 0)) for s in sorted(arm_count)}
    return th, ess, support, folded


# Arms

def collect_arms(deg_dirs, target_order, out_root):
    '''(target, tool, mode) -> list of primers, for both run modes.'''
    arms = []
    for target in target_order:
        for mode, root in (('non-deg', NONDEG_DIRS.get(target)), ('deg', deg_dirs.get(target))):
            if not root:
                continue
            path = os.path.join(out_root, root, 'designed_primers.csv')
            if not os.path.isfile(path):
                print('  skipping %s (%s): no %s' % (target, mode, path))
                continue
            df = pd.read_csv(path)
            for tool in TOOLS:
                seqs = [str(s).upper() for s in df.loc[df['tool'] == tool, 'primer']]
                if seqs:
                    arms.append({'target': target, 'tool': tool, 'mode': mode,
                                 'run_dir': root, 'primers': seqs})
    return arms


def check_externals_identical(arms):
    '''The claim the two-mode comparison rests on: for the external tools,
    nothing but the Olivia arms differs between the two directories.'''
    by = collections.defaultdict(dict)
    for a in arms:
        by[(a['target'], a['tool'])][a['mode']] = set(a['primers'])
    bad = []
    for (target, tool), modes in sorted(by.items()):
        if tool.startswith('Olivia') or len(modes) < 2:
            continue
        if modes['deg'] != modes['non-deg']:
            bad.append('%s/%s' % (target, tool))
    if bad:
        print('\nWARNING: these external arms differ between the deg and non-deg '
              'runs, so they are not one row: %s' % ', '.join(bad))
    else:
        print('\nexternal tools identical across both run directories, as expected')
    return not bad


# Output

def write_tex(rates, target_order, path, quantile):
    '''A standalone table, in the shape of tables_viral_benchmark.tex but with
    the mode column and both threshold readings.'''
    name = {'SARS-CoV-2': 'SARS-CoV-2', 'Measles': 'Measles', 'BoDV-1': 'BoDV-1',
            'HAV': 'Hepatitis A', 'Polio': 'Poliovirus 1', 'Rabies': 'Rabies',
            'rat HEV': 'Rat hepatitis E', 'HIV-1': 'HIV-1',
            'HEV': 'Hepatitis E', 'HCV': 'Hepatitis C'}
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        "\\caption{\\textbf{Abnormality rates in both design modes.} Percentage of "
        "each scheme's primers flagged by at least one Olivia encoder head. "
        "\\textbf{Published} applies the checkpoint's own %dth-percentile "
        "thresholds; \\textbf{Matched} applies thresholds recalibrated on real "
        "primers reweighted to the arm's own degenerate-base profile. "
        "\\textbf{Deg.} is the percentage of the arm's primers carrying at least "
        "one degenerate base.}" % round(100 * quantile),
        r'\label{tab:abnormality_both_modes}', r'\vspace{6pt}',
        r'\begin{tabularx}{\textwidth}{llYYYY}', r'\toprule',
        r'\textbf{Viral Dataset} & \textbf{Tool (mode)} & \textbf{Deg.\ (\%)} & '
        r'\textbf{Published (\%)} $\downarrow$ & \textbf{Matched (\%)} $\downarrow$ & '
        r'\textbf{ESS} \\', r'\midrule', '',
    ]
    for target in target_order:
        sub = rates[rates['target'] == target]
        if sub.empty:
            continue
        lines.append('%% --- %s ---' % target)
        lines.append(r'\multirow{%d}{*}{\textbf{%s}}' % (len(sub), name.get(target, target)))
        for _, r in sub.iterrows():
            label = r['tool'] if r['tool'].startswith('Olivia') else r['tool']
            if r['mode_label']:
                label = '%s (%s)' % (label, r['mode_label'])
            lines.append(r'& %s & %.1f & %.2f & %.2f & %d \\'
                         % (label.replace('_', r'\_'), r['pct_primers_degenerate'],
                            r['abnormal_any_pct_published'],
                            r['abnormal_any_pct_matched'], round(r['ess_matched'])))
        lines.append(r'\midrule')
        lines.append('')
    if lines[-2] == r'\midrule':
        lines.pop(-2)
    lines += [r'\bottomrule', r'\end{tabularx}', r'\end{table}']
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def plot_overall(rates, target_order, stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    arms = [('Olivia', 'deg', 'Olivia-deg', '#2a78d6'),
            ('Olivia', 'non-deg', 'Olivia-nondeg', '#7fb2e8'),
            ('Olivar', 'non-deg', 'Olivar', '#d1752b'),
            ('PrimalScheme3', 'non-deg', 'PrimalScheme3', '#9b6fc4'),
            ('varVAMP', 'non-deg', 'varVAMP', '#2e9e8f')]
    targets = [t for t in target_order if (rates['target'] == t).any()]

    fig, ax = plt.subplots(figsize=(12, 4.2))
    width = 0.8 / len(arms)
    x = np.arange(len(targets))
    for i, (tool, mode, label, colour) in enumerate(arms):
        vals = []
        for t in targets:
            sel = rates[(rates['target'] == t) & (rates['tool'] == tool)
                        & (rates['mode'] == mode)]
            vals.append(float(sel['abnormal_any_pct_matched'].iloc[0]) if len(sel) else np.nan)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=label, color=colour)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=20, ha='right')
    ax.set_ylabel('Abnormality rate (%)' + r'  $\downarrow$')
    ax.set_title('Overall abnormality rate at degeneracy-matched thresholds, '
                 'Olivia in both design modes')
    ax.legend(frameon=False, ncol=5, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig('%s.%s' % (stem, ext), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved %s.{pdf,png,svg}' % stem)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(HERE, 'output', 'deg_abnormality'))
    ap.add_argument('--calib-dir', default=os.path.join(
        config.REPO_DIR, 'data', 'embedded', 'true_data_all_t37'),
        help='corpus to recalibrate on [the all-taxa t37 corpus the checkpoint was fitted on]')
    ap.add_argument('--splits', nargs='*', default=['val', 'test'],
                    help="held-out splits of that corpus to pool [val test]")
    ap.add_argument('--n-zero', type=int, default=30000,
                    help='non-degenerate primers to keep in the calibration pool [30000]')
    ap.add_argument('--strata-cap', type=int, default=6,
                    help='degenerate-base counts at or above this are one stratum [6]')
    ap.add_argument('--quantile', type=float, default=None,
                    help="flag quantile [the checkpoint's own]")
    ap.add_argument('--targets', nargs='*', default=None)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    out_root = os.path.join(HERE, 'output')

    deg_dirs, target_order = canonical_deg_dirs()
    if args.targets:
        unknown = [t for t in args.targets if t not in target_order]
        if unknown:
            raise SystemExit('unknown target(s): %s' % ', '.join(unknown))
        target_order = [t for t in target_order if t in args.targets]
    print('canonical deg runs (same per-target threshold as the mismatch figures):')
    for t in target_order:
        print('  %-11s %s' % (t, deg_dirs.get(t)))

    import torch
    from encoder_filter import EncoderScorer
    scorer = EncoderScorer(config.ENCODER_MODEL, threads=args.threads)
    ckpt = torch.load(config.ENCODER_MODEL, map_location='cpu', weights_only=False)
    published = {'A': ckpt['threshold_head_a'], 'C_marg': ckpt['threshold_head_c_marg'],
                 'C_inter': ckpt['threshold_head_c_inter'], 'P': ckpt['threshold_head_p']}
    quantile = args.quantile if args.quantile is not None else float(
        ckpt.get('threshold_quantile', 0.95))
    print('\nencoder: %s\npublished thresholds (q=%.2f): %s'
          % (config.ENCODER_MODEL, quantile,
             ', '.join('%s %.4f' % (h, published[h]) for h in HEADS)))

    print('\nreading calibration corpus %s' % args.calib_dir)
    pool_seqs, pool_ndeg, corpus_strata = load_calibration_pool(args, rng)
    pool_scores = score_pool(scorer, pool_seqs,
                             os.path.join(args.out, 'calibration_scores.npz'),
                             'calibration pool')

    # Method check.
    corpus_th, corpus_ess, _, _ = thresholds_for_histogram(
        corpus_strata, pool_ndeg, pool_scores, args.strata_cap, quantile)
    print('\ncorpus-reweighted q=%.2f vs the checkpoint (ESS %d):'
          % (quantile, round(corpus_ess)))
    worst = 0.0
    for h in HEADS:
        rel = abs(corpus_th[h] - published[h]) / max(abs(published[h]), 1e-12)
        worst = max(worst, rel)
        print('  %-8s reweighted %.4f   checkpoint %.4f   (%.1f%%)'
              % (h, corpus_th[h], published[h], 100 * rel))
    print('  largest relative gap %.1f%%%s' % (
        100 * worst,
        '' if worst < 0.05 else '  -- the calibration corpus does not match the '
        'checkpoint; pass --calib-dir for the right one'))
    pd.DataFrame([{'head': h, 'threshold_corpus_reweighted': corpus_th[h],
                   'threshold_checkpoint': published[h],
                   'corpus_ess': corpus_ess} for h in HEADS]).to_csv(
        os.path.join(args.out, 'calibration_check.csv'), index=False)

    arms = collect_arms(deg_dirs, target_order, out_root)
    check_externals_identical(arms)

    print('\nscoring %d arms' % len(arms))
    rows, per_primer, support_rows = [], [], []
    for a in arms:
        seqs = a['primers']
        ndeg = np.array([n_degenerate(s) for s in seqs])
        s = scorer.score(seqs)
        s = {h: np.asarray(s[h], dtype=np.float64) for h in HEADS}

        th, ess, support, folded = matched_thresholds(
            ndeg, pool_ndeg, pool_scores, args.strata_cap, quantile)

        rec = {'target': a['target'], 'tool': a['tool'], 'mode': a['mode'],
               'run_dir': a['run_dir'], 'n_primers': len(seqs),
               'pct_primers_degenerate': 100.0 * float((ndeg > 0).mean()),
               'mean_degenerate_bases': float(ndeg.mean()),
               'max_degenerate_bases': int(ndeg.max()),
               'ess_matched': ess}
        # An Olivia arm is the only one that exists in both modes; for the rest
        # the mode column would imply a contrast that is not there.
        rec['mode_label'] = a['mode'] if a['tool'].startswith('Olivia') else ''

        flags = {}
        for kind, thresholds in (('published', published), ('matched', th)):
            f = {h: s[h] > thresholds[h] for h in HEADS}
            anyf = np.any(np.stack([f[h] for h in HEADS]), axis=0)
            flags[kind] = anyf
            for h in HEADS:
                rec['abnormal_%s_pct_%s' % (h, kind)] = 100.0 * float(f[h].mean())
            rec['abnormal_any_pct_%s' % kind] = 100.0 * float(anyf.mean())
        for h in HEADS:
            rec['threshold_%s_matched' % h] = th[h]
        rec['strata_folded'] = ';'.join('%d->%d' % kv for kv in sorted(folded.items()))
        rows.append(rec)

        for st, n_ref in sorted(support.items()):
            support_rows.append({'target': a['target'], 'tool': a['tool'],
                                 'mode': a['mode'], 'stratum': st,
                                 'n_arm_primers': int((strata_of(ndeg, args.strata_cap) == st).sum()),
                                 'n_reference_primers': n_ref})

        for i, seq in enumerate(seqs):
            per_primer.append({
                'target': a['target'], 'tool': a['tool'], 'mode': a['mode'],
                'primer': seq, 'n_degenerate_bases': int(ndeg[i]),
                'A': s['A'][i], 'C_marg': s['C_marg'][i],
                'C_inter': s['C_inter'][i], 'P': s['P'][i],
                'abnormal_published': bool(flags['published'][i]),
                'abnormal_matched': bool(flags['matched'][i])})

        print('  %-11s %-14s %-8s n=%3d deg=%5.1f%%  published %5.2f%%  '
              'matched %5.2f%%  ESS %6d%s'
              % (a['target'], a['tool'], a['mode'], len(seqs),
                 rec['pct_primers_degenerate'], rec['abnormal_any_pct_published'],
                 rec['abnormal_any_pct_matched'], round(ess),
                 '  folded ' + rec['strata_folded'] if folded else ''))

    rates = pd.DataFrame(rows)
    order = {t: i for i, t in enumerate(target_order)}
    tool_order = {('Olivia', 'deg'): 0, ('Olivia', 'non-deg'): 1}
    rates['_t'] = rates['target'].map(order)
    rates['_a'] = [tool_order.get((r.tool, r.mode), 2 + TOOLS.index(r.tool))
                   for r in rates.itertuples()]
    rates = rates.sort_values(['_t', '_a']).drop(columns=['_t', '_a']).reset_index(drop=True)

    rates.to_csv(os.path.join(args.out, 'abnormality_both_modes.csv'), index=False)
    pd.DataFrame(per_primer).to_csv(
        os.path.join(args.out, 'per_primer_scores.csv'), index=False)
    pd.DataFrame(support_rows).to_csv(
        os.path.join(args.out, 'strata_support.csv'), index=False)
    write_tex(rates, target_order,
              os.path.join(args.out, 'table_abnormality_both_modes.tex'), quantile)
    plot_overall(rates, target_order,
                 os.path.join(args.out, 'abnormality_both_modes'))

    print('\n--- overall abnormality rate, matched thresholds ---')
    piv = rates.pivot_table(index=['tool', 'mode'], columns='target',
                            values='abnormal_any_pct_matched')
    piv = piv.reindex(columns=[t for t in target_order if t in piv.columns])
    print(piv.round(2).to_string())
    print('\nwrote %s' % args.out)


if __name__ == '__main__':
    main()
