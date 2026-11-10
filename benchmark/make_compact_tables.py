#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Supplementary mismatch and abnormality tables, Olivia in both modes.

    python make_compact_tables.py
'''

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mismatch_analysis as ma

LABELS = {'SARS-CoV-2': 'SARS-CoV-2', 'RSV-B': 'RSV-B', 'RSV-A': 'RSV-A',
          'Measles': 'MeV', 'BoDV-1': 'BoDV-1', 'HAV': 'HAV',
          'Rabies': 'RABV', 'Polio': 'PV-1', 'HIV-1': 'HIV-1', 'HEV': 'HEV',
          'rat HEV': 'Rat HEV', 'HCV': 'HCV'}

# (arm, column label, degeneracy class). Non-degenerate arms first; bold is
# computed within a class, never across the two.
MM_ARMS = [('Olivia-nondeg', 'Olivia', 'nondeg'),
           ('Olivar', 'Olivar', 'nondeg'),
           ('PrimalScheme3', 'PS3', 'nondeg'),
           ('Olivia-deg', 'Olivia', 'deg'),
           ('varVAMP', 'varVAMP', 'deg')]

AB_ARMS = [('Olivia-nondeg', 'Olivia', 'nondeg'),
           ('Olivar', 'Olivar', 'nondeg'),
           ('PrimalScheme3', 'PS3', 'nondeg'),
           ('Olivia-deg', 'Olivia', 'deg'),
           ('varVAMP', 'varVAMP', 'deg')]


def marked(row, arms, lower_better, nd):
    '''Format one metric across arms, bolding the best value in the row.'''
    vals = [row[a] for a, _l, _c in arms if not pd.isna(row.get(a))]
    best = (min(vals) if lower_better else max(vals)) if vals else None
    out = []
    for a, _l, _cls in arms:
        v = row.get(a)
        if pd.isna(v):
            out.append('--')
            continue
        s = '%.*f' % (nd, v)
        if best is not None and abs(v - best) < 5e-9:
            s = r'\textbf{%s}' % s
        out.append(s)
    return out


def plain(row, arms, nd):
    return ['--' if pd.isna(row.get(a)) else '%.*f' % (nd, row.get(a))
            for a, _l, _c in arms]


def group_header(arms, title):
    '''A metric block header split into its two degeneracy classes.'''
    n_nd = sum(1 for _a, _l, c in arms if c == 'nondeg')
    n_d = len(arms) - n_nd
    return (r'\multicolumn{%d}{c}{%s, non-deg.} & \multicolumn{%d}{c}{%s, deg.}'
            % (n_nd, title, n_d, title))


def mismatch_table(order):
    d = pd.read_csv(os.path.join(HERE, 'output', 'mismatch_both_modes', 'summary.csv'))
    arms = [a for a, _l, _c in MM_ARMS]
    d = d[d.arm.isin(arms)]
    have = [t for t in order if t in set(d['virus'])]
    mm = d.pivot(index='virus', columns='arm', values='mean_mismatches_per_primer')
    pp = d.pivot(index='virus', columns='arm', values='pct_pairs_perfect')
    dg = d.pivot(index='virus', columns='arm', values='pct_primers_degenerate')

    # Column headers are deliberately NOT bold: bold is reserved for the
    # best value in the body, so it carries one meaning only.
    hdr = ' & '.join(l for _a, l, _c in MM_ARMS)
    n = len(MM_ARMS)
    out = [r'\begin{table}[htbp]', r'\centering',
           r'\caption{\textbf{Primer mismatches against the input alignment, '
           r'with Olivia in both design modes.} Each primer of every scheme '
           r'was compared with every sequence of the alignment it was designed '
           r'from; a position counts as a mismatch only where no resolution of '
           r'the target base is compatible with any permutation of the primer. '
           r'Targets are ordered by mean per-column alignment entropy, most '
           r'conserved first. Arms are grouped by whether they emit degenerate '
           r'primers. Bold marks the best arm for each target, over all five '
           r'arms; because the metric is not invariant to degeneracy, the '
           r'degeneracy each arm spent is given in the last two columns.}',
           r'\label{tab:mismatch_compact}', r'\vspace{6pt}',
           r'\scriptsize', r'\renewcommand{\arraystretch}{1.1}',
           r'\setlength{\tabcolsep}{2.5pt}',
           r'\begin{tabular}{@{}l rrr rr rrr rr rr@{}}', r'\toprule',
           r' & \multicolumn{%d}{c}{\textbf{Mismatches per primer} $\downarrow$}'
           r' & \multicolumn{%d}{c}{\textbf{Pairs, 0 mismatches (\%%)} $\uparrow$}'
           r' & \multicolumn{2}{c}{\textbf{Degenerate (\%%)}} \\' % (n, n),
           r'\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}\cmidrule(lr){%d-%d}'
           % (1 + n, 2 + n, 1 + 2 * n, 2 + 2 * n, 3 + 2 * n),
           r'Virus & ' + hdr + ' & ' + hdr +
           r' & Olivia & varVAMP \\',
           r'\addlinespace[1pt]',
           r' & \multicolumn{3}{c}{\textit{non-deg.}} & \multicolumn{2}{c}{\textit{deg.}}'
           r' & \multicolumn{3}{c}{\textit{non-deg.}} & \multicolumn{2}{c}{\textit{deg.}}'
           r' & \multicolumn{2}{c}{\textit{deg.\ mode}} \\',
           r'\midrule']

    for t in have:
        # Fixed 3 decimal places.
        cells = marked(mm.loc[t], MM_ARMS, True, 3)
        cells += marked(pp.loc[t], MM_ARMS, False, 1)
        cells += ['%.1f' % dg.loc[t, 'Olivia-deg'], '%.1f' % dg.loc[t, 'varVAMP']]
        out.append('%s & %s \\\\' % (LABELS.get(t, t), ' & '.join(cells)))

    out.append(r'\midrule')
    mrow = mm.loc[have].mean()
    prow = pp.loc[have].mean()
    cells = marked(mrow, MM_ARMS, True, 3) + marked(prow, MM_ARMS, False, 1)
    cells += ['%.1f' % dg.loc[have, 'Olivia-deg'].mean(),
              '%.1f' % dg.loc[have, 'varVAMP'].mean()]
    out.append(r'\textit{Mean} & %s \\' % ' & '.join(cells))
    out += [r'\bottomrule', r'\end{tabular}', '',
            r'\vspace{6pt}', r'\begin{minipage}{\linewidth}', r'\scriptsize',
            r'\textit{Note:} PS3, PrimalScheme3. Within each metric the first '
            r'three columns are the non-degenerate arms (Olivia at '
            r'\texttt{deg=False}, Olivar, PrimalScheme3) and the last two the '
            r'degenerate arms (Olivia at \texttt{deg=True}, varVAMP). Bold '
            r'marks the best arm for that target over all five. Two '
            r'like-for-like readings are worth separating from it. Among the '
            r'non-degenerate '
            r'arms, which share the Olivar tiling engine and differ only in '
            r'candidate selection, Olivia is best on 10 of 11 targets '
            r'(mean 0.93 against 0.98 and 1.78). Among the degenerate arms '
            r'Olivia is best on 8 of 11 (mean 0.55 against 0.81), the '
            r'exceptions being HAV, RABV and PV-1, on which varVAMP carried '
            r'1.7--4.5$\times$ the degeneracy of Olivia. Because a mismatch '
            r'is forgiven wherever any permutation of the primer matches, '
            r'these metrics are not invariant to the last two columns, which '
            r'is why the two groups are not compared directly. '
            r'PrimalScheme3 reports 0.0\% degeneracy but handles variation by '
            r'emitting several discrete alternates at one site, a mechanism '
            r'this column does not measure.',
            r'\end{minipage}', r'\end{table}']
    return '\n'.join(out), have


def abnormality_table(order):
    d = pd.read_csv(os.path.join(HERE, 'output', 'deg_abnormality',
                                 'abnormality_both_modes.csv'))
    d = d[d.target.isin(order)].copy()
    # One row per (target, arm); external tools appear once.
    def arm_of(r):
        if r['tool'] == 'Olivia':
            return 'Olivia-deg' if r['mode'] == 'deg' else 'Olivia-nondeg'
        return r['tool']
    d['arm'] = d.apply(arm_of, axis=1)
    d = d[~(d['tool'].isin(('Olivar', 'PrimalScheme3', 'varVAMP'))
            & (d['mode'] == 'deg'))]

    arms = [a for a, _l, _c in AB_ARMS]
    have = [t for t in order if t in set(d['target'])]
    pub = d.pivot_table(index='target', columns='arm',
                        values='abnormal_any_pct_published')
    mat = d.pivot_table(index='target', columns='arm',
                        values='abnormal_any_pct_matched')
    ess = d.pivot_table(index='target', columns='arm', values='ess_matched')
    for f in (pub, mat, ess):
        for a in arms:
            if a not in f.columns:
                f[a] = float('nan')

    hdr = ' & '.join(l for _a, l, _c in AB_ARMS)
    n = len(AB_ARMS)
    out = [r'\begin{table}[htbp]', r'\centering',
           r'\caption{\textbf{Overall abnormality rate, with Olivia in both '
           r'design modes and at two threshold calibrations.} Percentage of '
           r'the primers in a scheme flagged by at least one of the four '
           r'Olivia encoder heads. \textbf{Published} uses the '
           r'95th-percentile thresholds carried by the checkpoint; '
           r'\textbf{Matched} uses thresholds recalibrated on real primers '
           r'reweighted to the degenerate-base profile of that arm. Lower is '
           r'better; bold marks the best arm for each target, over all six '
           r'arms within that threshold block. The '
           r'per-head breakdown is in '
           r'Table~\ref{tab:viral_benchmark_evaluation}.}',
           r'\label{tab:abnormality_compact}', r'\vspace{6pt}',
           r'\scriptsize', r'\renewcommand{\arraystretch}{1.1}',
           r'\setlength{\tabcolsep}{2.5pt}',
           r'\begin{tabular}{@{}l rrrr rr rrrr rr@{}}', r'\toprule',
           r' & \multicolumn{%d}{c}{\textbf{Published thresholds (\%%)} $\downarrow$}'
           r' & \multicolumn{%d}{c}{\textbf{Degeneracy-matched (\%%)} $\downarrow$} \\'
           % (n, n),
           r'\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}' % (1 + n, 2 + n, 1 + 2 * n),
           r'Virus & ' + hdr + ' & ' + hdr + r' \\',
           r'\addlinespace[1pt]',
           r' & \multicolumn{4}{c}{\textit{non-deg.}} & \multicolumn{2}{c}{\textit{deg.}}'
           r' & \multicolumn{4}{c}{\textit{non-deg.}} & \multicolumn{2}{c}{\textit{deg.}} \\',
           r'\midrule']

    for t in have:
        cells = marked(pub.loc[t], AB_ARMS, True, 2) + \
                marked(mat.loc[t], AB_ARMS, True, 2)
        out.append('%s & %s \\\\' % (LABELS.get(t, t), ' & '.join(cells)))

    out.append(r'\midrule')
    cells = marked(pub.loc[have].mean(), AB_ARMS, True, 2) + \
            marked(mat.loc[have].mean(), AB_ARMS, True, 2)
    out.append(r'\textit{Mean} & %s \\' % ' & '.join(cells))

    lo_od = ess.loc[have, 'Olivia-deg'].min()
    lo_vv = ess.loc[have, 'varVAMP'].min()
    out += [r'\bottomrule', r'\end{tabular}', '',
            r'\vspace{6pt}', r'\begin{minipage}{\linewidth}', r'\scriptsize',
            r'\textit{Note:} PS3, PrimalScheme3. Olivar and PrimalScheme3 emit '
            r'identical primers in both modes and are reported once. Two '
            r'readings are given because the published thresholds were '
            r'calibrated on a corpus that is 99.1\% non-degenerate, while a '
            r'degenerate primer is encoded as a probability matrix over the '
            r'bases its IUPAC code stands for. '
            r'The matched thresholds rest on thin '
            r'reference strata where degeneracy is heaviest -- the lowest '
            r'Kish effective sample size is ' + str(int(lo_od)) +
            r' for Olivia-deg and ' + str(int(lo_vv)) +
            r' for varVAMP, against 30{,}000 for a non-degenerate arm -- so the '
            r'matched column is the less precisely estimated of the two and '
            r'neither on its own settles a degenerate-against-non-degenerate '
            r'comparison.',
            r'\end{minipage}', r'\end{table}']
    return '\n'.join(out), have


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(HERE, 'output', 'tables_compact.tex'))
    args = ap.parse_args()

    order = ma.TARGET_ORDER
    mm, mm_have = mismatch_table(order)
    ab, ab_have = abnormality_table(order)
    with open(args.out, 'w') as fh:
        fh.write(mm + '\n\n' + ab + '\n')
    print('mismatch    %2d targets: %s' % (len(mm_have), ', '.join(mm_have)))
    print('abnormality %2d targets: %s' % (len(ab_have), ', '.join(ab_have)))
    print('\n-> %s' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
