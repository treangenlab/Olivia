#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Supplementary breadth / recovery tables.

    python make_design_tables.py
'''

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench_targets
import mismatch_analysis as ma
import msa_stats as M

# Display order: mean per-column alignment entropy, most conserved first, the
# same order the figures and the compact tables use.
ORDER = ['SARS-CoV-2', 'RSV-B', 'RSV-A', 'Measles', 'BoDV-1', 'HAV',
         'Rabies', 'Polio', 'HIV-1', 'HEV', 'rat HEV', 'HCV']

# analysis name -> bench_targets name, where the two registries disagree.
BT_NAME = {'BoDV-1': 'BDV-1', 'Polio': 'Polio-1', 'rat HEV': 'ratHEV'}

LABELS = {'SARS-CoV-2': 'SARS-CoV-2', 'RSV-B': 'RSV-B', 'RSV-A': 'RSV-A',
          'Measles': 'MeV', 'BoDV-1': 'BoDV-1', 'HAV': 'HAV',
          'Rabies': 'RABV', 'Polio': 'PV-1', 'HIV-1': 'HIV-1', 'HEV': 'HEV',
          'rat HEV': 'Rat HEV', 'HCV': 'HCV'}

# Arms of the breadth table, in row order.
ARMS = ['Olivia', 'Olivia-deg', 'Olivar', 'PrimalScheme3', 'varVAMP']


def bt(name):
    return bench_targets.BY_NAME[BT_NAME.get(name, name)]


def deg_target(name):
    '''A Target standing for the degenerate run of `name`, or None.'''
    entry = ma.TARGETS.get(name)
    if entry is None:
        return None
    msa_path, deg_dir = entry
    t = bench_targets.Target(name, os.path.basename(deg_dir), msa_path)
    return t if os.path.isfile(t.primer_table) else None


def var_pct(sub):
    '''% of placed sites at which this arm acted on sequence variation.'''
    if not len(sub):
        return float('nan')
    hit = 0
    for _site, rows in sub.groupby('site'):
        seqs = rows['primer'].astype(str).str.upper()
        degenerate = seqs.str.contains('[^ACGT]', regex=True).any()
        alternates = seqs.nunique() > 1
        hit += bool(degenerate or alternates)
    return 100.0 * hit / sub['site'].nunique()


def collect(names, ambiguous, cache):
    '''{target: {arm: row}} plus {target: alignment facts}.'''
    arms, aln = {}, {}
    for name in names:
        t = bt(name)
        if not os.path.isfile(t.primer_table):
            print('  %-12s no designed_primers.csv -- skipped' % name)
            continue
        print('  %-12s placing primers' % name)
        cons = M.consensus_len_of(t)
        msa = M.Msa(t.msa_path)
        stats = msa.column_stats()
        ident = msa.pairwise_identity()
        # pairwise_identity returns (mean, sd, n_pairs); column_stats names the
        # conservation share pct_conserved_95.
        aln[name] = {'n_seq': msa.n_seq, 'consensus_nt': cons,
                     'pct_identity': ident[0], 'pct_identity_sd': ident[1],
                     'pct_cons95': stats['pct_conserved_95']}

        rows = {}
        geom = M.place_primers(t, cache=cache, ambiguous=ambiguous, verbose=False)
        for tool, sub in geom.groupby('tool'):
            rec = M.summarise_arm(sub, cons)
            rec['var_pct'] = var_pct(sub)
            rows[tool] = rec

        dt = deg_target(name)
        if dt is not None:
            dgeom = M.place_primers(dt, cache=cache, ambiguous=ambiguous,
                                    verbose=False)
            sub = dgeom[dgeom['tool'] == 'Olivia']
            if len(sub):
                rec = M.summarise_arm(sub, cons)
                rec['var_pct'] = var_pct(sub)
                rows['Olivia-deg'] = rec
        else:
            print('  %-12s no deg run -- Olivia-deg row omitted' % name)
        arms[name] = rows
    return arms, aln


# LaTeX

def design_summary(names, arms, aln):
    out = [r'\begin{table}[htbp]', r'\centering',
           r'\caption{\textbf{Summary of Olivia designs for tiled sequencing '
           r'without degenerate bases.}}',
           r'\label{tab:olivia_design_summary}', r'\vspace{6pt}',
           r'\footnotesize', r'\renewcommand{\arraystretch}{1.1}',
           r'\setlength{\tabcolsep}{4pt}',
           r'\begin{tabular}{@{}lrcrrrr@{}}', r'\toprule',
           r' & \multicolumn{3}{c}{\textbf{Alignment}} '
           r'& \multicolumn{3}{c}{\textbf{Olivia design}} \\',
           r'\cmidrule(lr){2-4}\cmidrule(l){5-7}',
           r'\textbf{Virus} & \textbf{Seqs} & \textbf{Identity} & '
           r'\textbf{Cons.} & \textbf{Recov.} & \textbf{Amp.} & '
           r'\textbf{Primers} \\',
           r' & & (\%) & (\%) & (\%) & & \\', r'\midrule']
    for n in names:
        if n not in aln or 'Olivia' not in arms.get(n, {}):
            continue
        a, o = aln[n], arms[n]['Olivia']
        out.append(r'%-10s & %d & $%.1f \pm %.1f$ & %.1f & %.1f & %d & %d \\'
                   % (LABELS[n], a['n_seq'], a['pct_identity'],
                      a['pct_identity_sd'], a['pct_cons95'],
                      o['pct_insert_coverage'], o['n_amplicons'],
                      o['n_primers']))
    out += [r'\bottomrule', r'\end{tabular}', '',
            r'\vspace{6pt}', r'\begin{minipage}{\linewidth}', r'\scriptsize',
            r'\textit{Note:} SARS-CoV-2, severe acute respiratory syndrome '
            r'coronavirus~2; RSV-A/RSV-B, respiratory syncytial virus '
            r'subgroups~A and~B; MeV, measles virus; BoDV-1, Borna disease '
            r'virus~1; HAV, hepatitis~A virus; RABV, rabies virus; PV-1, '
            r'poliovirus~1; HIV-1, human immunodeficiency virus~1; HEV, '
            r'hepatitis~E virus; Rat HEV, rat hepatitis~E virus; HCV, '
            r'hepatitis~C virus. Targets are ordered by mean per-column '
            r'alignment entropy, most conserved first. \textit{Seqs}: '
            r'sequences in the alignment. \textit{Identity}: mean $\pm$ SD of '
            r'pairwise sequence identity, skipping gaps and ambiguity codes. '
            r'\textit{Cons.}: share of alignment columns with gap frequency '
            r'$<0.5$ whose majority base reaches $95\%$. \textit{Recov.}: '
            r'expected recovery, the union of primer-trimmed amplicon inserts '
            r'over the gap-masked majority consensus. \textit{Amp.}: '
            r'amplicons in the scheme. This table describes the '
            r'non-degenerate arm; the degenerate arm is in '
            r'Table~\ref{tab:viral_benchmark_breadth}.',
            r'\end{minipage}', r'\end{table}']
    return '\n'.join(out)


BREADTH_NOTE_BASE = (
    r'\textbf{Breadth} is the union of amplicon spans over the '
    r'gap-masked majority consensus, \textbf{Recov.} the union of the '
    r'primer-trimmed inserts -- what a run actually reads back -- and '
    r'\textbf{Span} the distance from the first to the last amplicon, '
    r'which separates a scheme that tiles the whole alignment thinly '
    r'from one that tiles part of it densely. Widest breadth and '
    r'highest recovery per target in bold. Olivia is the '
    r'non-degenerate arm and Olivia-deg the same pipeline with '
    r'degenerate primers enabled at the per-target consensus threshold '
    r'of Table~\ref{tab:benchmark_target_parameters}, the same runs as '
    r'Figs.~\ref{fig:mismatch_by_position} '
    r'and~\ref{fig:mismatch_summary}.')

# Only used when the Var. column is switched back on with --with-var.
BREADTH_NOTE_VAR = (
    r' \textbf{Var.} is the share of placed binding sites at which a '
    r'tool acted on sequence variation, put on one scale across the '
    r'two mechanisms in use: a primer carrying at least one '
    r'degenerate base, for the tools that emit one primer per site, '
    r'or two or more discrete alternates at the same site, for '
    r'PrimalScheme3, whose primers carry no degenerate bases at all. '
    r'A template amplifies if any alternate at a site matches it, '
    r'which is what makes the two one quantity. Olivia and Olivar are '
    r'run without degeneracy here and so sit at zero by construction.')


def breadth_one(names, arms, label, caption, note, with_var=False):
    '''One breadth table over `names`.'''
    out = [r'\begin{table}[htbp]', r'\centering',
           r'\caption{%s}' % caption,
           r'\label{%s}' % label, r'\vspace{6pt}',
           r'\footnotesize', r'\renewcommand{\arraystretch}{1.05}',
           r'\setlength{\tabcolsep}{4pt}',
           r'\begin{tabular}{@{}llrrrr%s@{}}' % ('r' if with_var else ''),
           r'\toprule',
           r'\textbf{Virus} & \textbf{Tool} & \textbf{Amp.} & '
           r'\textbf{Breadth} & \textbf{Recov.} & \textbf{Span}%s \\'
           % (r' & \textbf{Var.}' if with_var else ''),
           r' & & & (\%) & (\%) & (\%)' + (r' & (\%)' if with_var else '')
           + r' \\', r'\midrule']
    blocks = []
    for n in names:
        rows = arms.get(n, {})
        present = [a for a in ARMS if a in rows]
        if not present:
            continue
        # Bold the widest breadth and the highest recovery in the block.
        best_b = max(rows[a]['pct_coverage'] for a in present)
        best_r = max(rows[a]['pct_insert_coverage'] for a in present)
        blk = [r'%% --- %s ---' % n,
               r'\multirow{%d}{*}{\textbf{%s}}' % (len(present), LABELS[n])]
        for a in present:
            r_ = rows[a]
            b = ('\\textbf{%.1f}' % r_['pct_coverage']
                 if abs(r_['pct_coverage'] - best_b) < 5e-9
                 else '%.1f' % r_['pct_coverage'])
            rc = ('\\textbf{%.1f}' % r_['pct_insert_coverage']
                  if abs(r_['pct_insert_coverage'] - best_r) < 5e-9
                  else '%.1f' % r_['pct_insert_coverage'])
            blk.append(r'& %s & %d & %s & %s & %.1f%s \\'
                       % (a, r_['n_amplicons'], b, rc, r_['pct_span'],
                          (' & %.1f' % r_['var_pct']) if with_var else ''))
        blocks.append('\n'.join(blk))
    out.append('\n\\midrule\n'.join(blocks))
    out += [r'\bottomrule', r'\end{tabular}']
    if note:
        out += ['', r'\vspace{6pt}', r'\begin{minipage}{\linewidth}',
                r'\scriptsize', note, r'\end{minipage}']
    out.append(r'\end{table}')
    return '\n'.join(out)


def breadth(names, arms, split=None, with_var=False):
    '''The breadth table, split across two tables so each fits a page.'''
    k = split or (len(names) + 1) // 2
    first, rest = names[:k], names[k:]
    note = BREADTH_NOTE_BASE + (BREADTH_NOTE_VAR if with_var else '')
    cap = (r'\textbf{Amplicon count, genome breadth and recovery for every '
           r'tool on every target.} Reported so that the abnormality rates of '
           r'Fig.~\ref{fig:viral_benchmark_abnormality_overall} can be read '
           r'against the amount of genome each scheme actually commits to '
           r'covering.')
    if not rest:
        return breadth_one(first, arms, 'tab:viral_benchmark_breadth',
                           cap, note, with_var)
    a = breadth_one(
        first, arms, 'tab:viral_benchmark_breadth',
        cap + r' Continued in Table~\ref{tab:viral_benchmark_breadth_cont}; '
              r'see its note for column definitions.', None, with_var)
    b = breadth_one(
        rest, arms, 'tab:viral_benchmark_breadth_cont',
        r'\textbf{Amplicon count, genome breadth and recovery for every tool '
        r'on every target (continued).}', note, with_var)
    return a + '\n\n' + b


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--targets', nargs='*', default=None)
    ap.add_argument('--out', default=os.path.join(HERE, 'output', 'tables_design.tex'))
    ap.add_argument('--ambiguous', choices=M.AMBIGUOUS_MODES, default='resolve')
    ap.add_argument('--with-var', action='store_true',
                    help='restore the Var. column (share of sites at which the '
                         'arm acted on sequence variation); off by default '
                         'because the compact mismatch table already reports '
                         'degeneracy per arm')
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()

    names = args.targets or ORDER
    arms, aln = collect(names, args.ambiguous, not args.no_cache)
    have = [n for n in names if n in arms]
    if not have:
        raise SystemExit('no target produced rows')

    with open(args.out, 'w') as fh:
        fh.write(design_summary(have, arms, aln) + '\n\n'
                 + breadth(have, arms, with_var=args.with_var) + '\n')
    print('\n%d targets -> %s' % (len(have), args.out))
    for n in have:
        print('  %-12s arms: %s' % (n, ', '.join(a for a in ARMS
                                                 if a in arms[n])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
