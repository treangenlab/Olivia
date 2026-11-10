#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Mismatch summary figure with Olivia in both design modes.

    python plot_mismatch_summary_both_modes.py --degeneracy
'''

import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# Targets from most to least conserved; only those in summary.csv are drawn.
ORDER = ['SARS-CoV-2', 'RSV-B', 'RSV-A', 'Measles', 'BoDV-1', 'HAV',
         'Rabies', 'Polio', 'HIV-1', 'HEV', 'rat HEV', 'HCV']
LABELS = {'SARS-CoV-2': 'SARS-CoV-2', 'RSV-B': 'RSV-B', 'RSV-A': 'RSV-A',
          'Measles': 'Measles', 'BoDV-1': 'BoDV-1', 'HAV': 'HAV',
          'Rabies': 'RABV', 'Polio': 'PV-1', 'HIV-1': 'HIV-1', 'HEV': 'HEV',
          'rat HEV': 'Rat HEV', 'HCV': 'HCV'}

# Narrowed by main() to the targets present in the summary.csv it reads.
TARGETS = list(ORDER)

# Fixed order and fixed hues; the four published arms keep their published
# colours. See the module docstring for the validator result.
TOOLS = [('Olivia-nondeg', '#56B4E9'),
         ('Olivia-deg',    '#D55E00'),
         ('Olivar',        '#0072B2'),
         ('PrimalScheme3', '#E69F00'),
         ('varVAMP',       '#009E73')]
INK, GRID = '#1a1a1a', '#e6e6e6'
MM = 1 / 25.4
GROUP_W = 0.8


def style():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.edgecolor': INK, 'axes.labelcolor': INK,
        'xtick.color': INK, 'ytick.color': INK, 'text.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.dpi': 600,
    })


def bar_panel(ax, piv, ylabel, ticklabels=True, tools=None):
    tools = TOOLS if tools is None else tools
    x = np.arange(len(TARGETS))
    w = GROUP_W / len(tools)
    for k, (arm, col) in enumerate(tools):
        off = (k - (len(tools) - 1) / 2) * w
        ax.bar(x + off, piv[arm].values, width=w, color=col, edgecolor='white',
               linewidth=0.4, label=arm, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[t] for t in TARGETS] if ticklabels else [],
                       rotation=40, ha='right', rotation_mode='anchor')
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.5, len(TARGETS) - 0.5)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--summary', default=os.path.join(
        HERE, 'output', 'mismatch_both_modes', 'summary.csv'))
    ap.add_argument('--out', default=os.path.join(
        os.path.dirname(HERE), 'figures', 'mismatch_summary'))
    ap.add_argument('--no-panel-labels', action='store_true',
                    help='omit the internal a/b panel labels for composite figures')
    ap.add_argument('--degeneracy', action='store_true',
                    help='add a top panel with the percentage of degenerate primers '
                         'for the two arms that emit them (Olivia-deg, varVAMP); this '
                         'is the paper figure degenerate_percentage_mismatch_summary')
    args = ap.parse_args()

    d = pd.read_csv(args.summary)
    arms = [t[0] for t in TOOLS]
    d = d[d['arm'].isin(arms)]

    global TARGETS
    have = set(d['virus'])
    TARGETS = [t for t in ORDER if t in have]
    skipped = [t for t in ORDER if t not in have]
    extra = sorted(have - set(ORDER))
    if skipped:
        print('not in %s, omitted: %s'
              % (os.path.basename(args.summary), ', '.join(skipped)))
    if extra:
        # Appended rather than dropped silently; its place in the entropy
        # order is not known here.
        print('present but unranked, appended: %s' % ', '.join(extra))
        TARGETS += extra
    if not TARGETS:
        raise SystemExit('no known target in %s' % args.summary)

    missing = [(v, a) for v in TARGETS for a in arms
               if not ((d['virus'] == v) & (d['arm'] == a)).any()]
    if missing:
        raise SystemExit('summary.csv is missing %d (target, arm) cells, e.g. %s; '
                         'rerun mismatch_both_modes.py for them'
                         % (len(missing), ', '.join('%s/%s' % m for m in missing[:5])))

    mm = d.pivot(index='virus', columns='arm',
                 values='mean_mismatches_per_primer').loc[TARGETS]
    pp = d.pivot(index='virus', columns='arm',
                 values='pct_pairs_perfect').loc[TARGETS]

    style()
    # Drawn at the printed width (single-column \textwidth, 131 mm) so the
    # font sizes in style() are the sizes on the page.
    if args.degeneracy:
        dg = d.pivot(index='virus', columns='arm',
                     values='pct_primers_degenerate').loc[TARGETS]
        fig, (ax_d, ax_a, ax_b) = plt.subplots(3, 1, figsize=(131 * MM, 135 * MM),
                                               sharex=True,
                                               gridspec_kw=dict(hspace=0.16))
        fig.subplots_adjust(left=0.115, right=0.99, top=0.93, bottom=0.1)
        bar_panel(ax_d, dg, 'Degenerate\nprimers (%)', ticklabels=False,
                  tools=[t for t in TOOLS if t[0] in ('Olivia-deg', 'varVAMP')])
        ax_d.set_ylim(0, 100)
        ax_d.set_yticks([0, 25, 50, 75, 100])
    else:
        fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(131 * MM, 95 * MM), sharex=True,
                                         gridspec_kw=dict(hspace=0.14))
        fig.subplots_adjust(left=0.115, right=0.99, top=0.9, bottom=0.14)

    bar_panel(ax_a, mm, 'Mismatches\nper primer', ticklabels=False)
    top = float(np.ceil(mm.values.max()))
    ax_a.set_ylim(0, max(5.0, top))
    ax_a.set_yticks(np.arange(0, max(5.0, top) + 0.1, 1))
    bar_panel(ax_b, pp, 'Pairs with 0\nmismatches (%)')
    ax_b.set_ylim(0, 100)
    ax_b.set_yticks([0, 25, 50, 75, 100])

    if not args.no_panel_labels:
        panels = ((ax_d, 'a'), (ax_a, 'b'), (ax_b, 'c')) if args.degeneracy \
            else ((ax_a, 'a'), (ax_b, 'b'))
        for ax, letter in panels:
            ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontsize=9,
                    fontweight='bold', va='bottom', ha='left')

    fig.align_ylabels()
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(TOOLS), frameon=False,
               bbox_to_anchor=(0.55, 1.0), handlelength=1.0, handleheight=0.8,
               handletextpad=0.4, columnspacing=1.0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig('%s.%s' % (args.out, ext))
    print('wrote %s.{pdf,png,svg}' % args.out)
    print('\npanel a, mean mismatches per primer:')
    print(mm[arms].round(3).to_string())


if __name__ == '__main__':
    main()
