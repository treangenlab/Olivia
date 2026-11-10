#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Primer-vs-alignment mismatches per tool and target.

    python mismatch_analysis.py --out output/mismatch
    python mismatch_analysis.py --msa targets/hev/hev_stratified_aligned.fasta
    --virus HEV \\
        --primers Olivia-deg=output/HEV_deg/Olivia/amp400/design.csv --out
        output/HEV_deg/mismatch
'''

import argparse
import collections
import os
import sys

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.Align.substitution_matrices import Array

import config

# msa_tools and friends import each other flatly, so the package directory
# itself goes on the path, not just src/.
sys.path.insert(0, config.SRC_DIR)
sys.path.insert(0, config.PACKAGE_DIR)
from msa_tools import MSA, AMBREP, AMBMAPPING


# Degeneracy-aware local aligner (+1 where an IUPAC code covers the base, -1 elsewhere).
ALPHABET = 'ATCGUN-RYKMSWBDHV'
_SUBMTX = Array(alphabet=ALPHABET, dims=2) - 1
for _b1, _b2 in AMBMAPPING:
    _SUBMTX[_b1, _b2] = 1
    _SUBMTX[_b2, _b1] = 1
ALIGNER = PairwiseAligner(mode='local', open_gap_score=-2, extend_gap_score=-1,
                          substitution_matrix=_SUBMTX)


# IUPAC code -> the set of concrete bases it stands for.
RESOLVE = {b: {b} for b in 'ACGT'}
RESOLVE.update({k: set(v) for k, v in AMBREP.items()})
RESOLVE['U'] = {'T'}

COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', '-': '-', 'U': 'A',
              'N': 'N', 'R': 'Y', 'Y': 'R', 'K': 'M', 'M': 'K',
              'S': 'S', 'W': 'W', 'B': 'V', 'V': 'B', 'D': 'H', 'H': 'D'}

CONCRETE = np.array(list('ACGT-'))


# name -> (alignment, directory holding that target's designed_primers.csv).
TARGETS = {
    'SARS-CoV-2': (os.path.join(config.TARGETS_DIR, 'sars2', 'sars2_stratified_aligned.fasta'),
                   os.path.join(config.BENCH_DIR, 'output', 'SARS2_deg')),
    'Measles':    (os.path.join(config.TARGETS_DIR, 'measles', 'measles_stratified_aligned.fasta'),
                   os.path.join(config.BENCH_DIR, 'output', 'MEASLES_deg')),
    'BoDV-1':     (os.path.join(config.TARGETS_DIR, 'bdv1', 'bdv1_stratified_aligned.fasta'),
                   os.path.join(config.BENCH_DIR, 'output', 'BDV1_deg_t0.9')),
    'HAV':        (os.path.join(config.TARGETS_DIR, 'hav', 'hav_stratified_aligned.fasta'),
                   os.path.join(config.BENCH_DIR, 'output', 'HAV_deg_t0.75')),
    'Polio':      (os.path.join(config.TARGETS_DIR, 'polio', 'polio_stratified_aligned.fasta'),
                   os.path.join(config.BENCH_DIR, 'output', 'POLIO1_deg_t0.75')),
    'Rabies':     (os.path.join(config.TARGETS_DIR, 'rabies', 'rabies_stratified_aligned.fasta'),
                   os.path.join(config.BENCH_DIR, 'output', 'RABIES_deg')),
    'rat HEV':    (os.path.join(config.TARGETS_DIR, 'rathev', 'rathev_stratified_aligned.fasta'),
                   os.path.join(config.BENCH_DIR, 'output', 'ratHEV_deg')),
    'HIV-1':      (os.path.join(config.TARGETS_DIR, 'hiv1', 'hiv1_stratified_aligned.fasta'),
                   os.path.join(config.BENCH_DIR, 'output', 'HIV1_deg')),
    'HEV':        (os.path.join(config.TARGETS_DIR, 'hev', 'hev_stratified_aligned.fasta'),
                   os.path.join(config.BENCH_DIR, 'output', 'HEV_deg')),
    'HCV':        (os.path.join(config.TARGETS_DIR, 'hcv', 'hcv_stratified_aligned.fasta'),
                   os.path.join(config.BENCH_DIR, 'output', 'HCV_deg')),
    'RSV-A':      (os.path.join(config.TARGETS_DIR, 'rsva_gisaid', 'rsva_stratified_aligned.fasta'),
                   os.path.join(config.BENCH_DIR, 'output', 'RSVA_deg')),
    'RSV-B':      (os.path.join(config.TARGETS_DIR, 'rsvb_gisaid', 'rsvb_stratified_aligned.fasta'),
                   os.path.join(config.BENCH_DIR, 'output', 'RSVB_deg')),
}
# Targets from most to least conserved (mean alignment column entropy).
TARGET_ORDER = ['SARS-CoV-2', 'RSV-B', 'RSV-A', 'Measles', 'BoDV-1', 'HAV',
                'Rabies', 'Polio', 'HIV-1', 'HEV', 'rat HEV', 'HCV']

# Panel order.
ARM_ORDER = ['Olivia-deg', 'Olivar', 'PrimalScheme3', 'varVAMP']


# Readers -- one per output format the benchmark produces

def read_olivar_design(path):
    '''Olivar/Olivia design.csv: one row per amplicon, fP and rP in columns.'''
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        amp = str(r['amplicon_id'])
        rows.append((amp + '_LEFT', 'forward', str(r['fP']).upper()))
        rows.append((amp + '_RIGHT', 'reverse', str(r['rP']).upper()))
    return rows


def read_bed7(path):
    '''PrimalScheme3 primer.bed / varVAMP primers.bed: 7-column BED.'''
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            f = line.rstrip('\n').split('\t')
            if len(f) < 7:
                continue
            rows.append((f[3], 'forward' if f[5] == '+' else 'reverse', f[6].upper()))
    return rows


def read_collected(path):
    '''benchmark designed_primers.csv -- several tools in one table.'''
    df = pd.read_csv(path)
    out = {}
    for tool, sub in df.groupby('tool'):
        out[str(tool)] = [(str(r['name']), str(r['direction']), str(r['primer']).upper())
                          for _, r in sub.iterrows()]
    return out


def load_primers(label, path):
    '''Returns {label: [(name, direction, seq), ...]}, sniffing the format.'''
    if path.endswith('.bed'):
        return {label: read_bed7(path)}
    head = pd.read_csv(path, nrows=1)
    if 'tool' in head.columns and 'primer' in head.columns:
        tables = read_collected(path)
        if label:   # an explicit label selects one tool out of the table
            return {label: tables[label]} if label in tables else tables
        return tables
    return {label: read_olivar_design(path)}


# Locating and scoring

def locate(primer, consensus):
    '''Primer -> (start, stop, strand) on the gapless consensus.'''
    fw = ALIGNER.align(primer, consensus)[0]
    rv = ALIGNER.align(primer, consensus, strand='-')[0]
    aln = max(fw, rv, key=lambda a: a.score)
    left = int(aln.coordinates[0][0])
    right = len(primer) - int(aln.coordinates[0][-1])
    c1 = int(aln.coordinates[1][0])
    c2 = int(aln.coordinates[1][-1])
    if c1 <= c2:
        return c1 - left, c2 + right, '+', aln.score
    return c2 - right, c1 + left, '-', aln.score


AMBIGUOUS_MODES = ('resolve', 'drop', 'match', 'mismatch')


def score_primer(primer, msa, start, stop, strand, ambiguous='resolve'):
    '''(mismatch, usable) matrices (n_sequences, len(primer)), primer 5'->3'; a position
    mismatches when the primer's and target's IUPAC sets are disjoint.
    '''
    if ambiguous not in AMBIGUOUS_MODES:
        raise ValueError('ambiguous must be one of %s, got %r'
                         % (', '.join(AMBIGUOUS_MODES), ambiguous))
    cols = [msa._consensus2msa[p] for p in range(start, stop)]
    sub = msa.matrix[:, cols]
    if strand == '-':
        sub = sub[:, ::-1]
        sub = np.vectorize(lambda b: COMPLEMENT.get(b, 'N'))(sub)

    n_seq, n_pos = sub.shape
    mismatch = np.zeros((n_seq, n_pos), dtype=bool)
    usable = np.ones((n_seq, n_pos), dtype=bool)
    for i in range(n_pos):
        col = sub[:, i]
        allowed = RESOLVE.get(primer[i], set('ACGT'))
        # one entry per distinct character in the column, then broadcast back
        uniq, inv = np.unique(col, return_inverse=True)
        # a gap resolves to nothing, so it is disjoint from every primer base
        # and stays a mismatch in all four modes
        tsets = [RESOLVE.get(u, set()) for u in uniq]
        disjoint = np.array([allowed.isdisjoint(t) for t in tsets])[inv]
        amb = np.array([len(t) > 1 for t in tsets])[inv]
        if ambiguous == 'drop':
            usable[:, i] = ~amb
            mismatch[:, i] = disjoint & ~amb
        elif ambiguous == 'match':
            mismatch[:, i] = disjoint & ~amb
        elif ambiguous == 'mismatch':
            mismatch[:, i] = disjoint | amb
        else:                                 # 'resolve'
            mismatch[:, i] = disjoint
    return mismatch, usable


def analyse(virus, label, primers, msa, min_score_frac, ambiguous='resolve'):
    '''One row per (primer, sequence) pair, plus a per-position tally.'''
    pair_rows = []
    pos_hits = collections.Counter()     # distance from 3' end -> n mismatching pairs
    pos_total = collections.Counter()    # distance from 3' end -> n pairs scored
    skipped = []

    n_cons = len(msa.consensus)
    for name, direction, seq in primers:
        seq = seq.upper().replace('U', 'T')
        if len(seq) < 2 or not set(seq) <= set(RESOLVE):
            skipped.append((name, 'unreadable sequence'))
            continue
        start, stop, strand, score = locate(seq, msa.consensus)
        if score < min_score_frac * len(seq):
            skipped.append((name, 'weak alignment (score %.1f over %d nt)' % (score, len(seq))))
            continue
        if start < 0 or stop > n_cons or stop - start != len(seq):
            skipped.append((name, 'indel or off-end placement (%d-%d for %d nt)'
                            % (start, stop, len(seq))))
            continue
        # The BED strand is authoritative where the readers give one; the
        # aligner's call is a cross-check, not a correction.
        if direction.startswith('rev'):
            strand = '-'
        elif direction.startswith('for'):
            strand = '+'

        mismatch, usable = score_primer(seq, msa, start, stop, strand, ambiguous)
        full = usable.all(axis=1)            # pairs with no ambiguous target base
        if not full.any():
            skipped.append((name, 'every target row ambiguous under the primer'))
            continue
        counts = mismatch[full].sum(axis=1)
        n_perm = int(np.prod([len(RESOLVE[b]) for b in seq]))
        for c in counts:
            pair_rows.append((virus, label, name, strand, len(seq), n_perm, int(c)))

        n_pos = mismatch.shape[1]
        for i in range(n_pos):
            d = n_pos - 1 - i                # 0 = 3'-most base
            pos_total[d] += int(full.sum())
            pos_hits[d] += int(mismatch[full, i].sum())

    pairs = pd.DataFrame(pair_rows, columns=[
        'virus', 'arm', 'primer_name', 'strand', 'primer_len',
        'n_permutations', 'n_mismatch'])
    positions = pd.DataFrame(
        [(virus, label, d, pos_hits[d], pos_total[d],
          100.0 * pos_hits[d] / pos_total[d])
         for d in sorted(pos_total) if pos_total[d]],
        columns=['virus', 'arm', 'dist_from_3prime', 'n_mismatch', 'n_scored',
                 'pct_mismatch'])
    return pairs, positions, skipped


# Figures

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4',
          '#008300', '#4a3aa7', '#e34948']
# One fixed colour per virus, keyed by name so a virus keeps its colour when --targets
# drops others.
VIRUS_COLOURS = {
    'SARS-CoV-2': '#3057B1',   # blue
    'Measles':    '#DD9314',   # amber
    'BoDV-1':     '#0C7219',   # green
    'HAV':        '#D76074',   # rose
    'Polio':      '#85398C',   # plum
    'Rabies':     '#1ABCB8',   # teal
    'rat HEV':    '#A03313',   # rust
    'HIV-1':      '#3D8FE4',   # sky
    'HEV':        '#699E32',   # olive
    'HCV':        '#C487E8',   # lavender
}


def _frame(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=1.0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)


def _panel_grid(n, ncols, width_per, height_per):
    '''n panels wrapped over ncols columns, sharing one y axis.'''
    import matplotlib.pyplot as plt
    nrows = int(np.ceil(n / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width_per * ncols + 1.6,
                                      height_per * nrows + 0.8),
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    flat = [ax for row in axes for ax in row]
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n], (nrows, ncols)


def _is_bottom(i, n, shape):
    '''True if panel i has no visible panel beneath it.'''
    _, ncols = shape
    return i + ncols >= n


def fig4a(pairs, arms, viruses, out_base, ncols=3):
    '''Mismatch count per primer-sequence pair, one panel per tool.'''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tbl = pairs.groupby(['arm', 'virus', 'n_mismatch']).size().rename('n_pairs').reset_index()
    tot = pairs.groupby(['arm', 'virus']).size().rename('n_total')
    tbl = tbl.join(tot, on=['arm', 'virus'])
    tbl['pct'] = 100.0 * tbl['n_pairs'] / tbl['n_total']
    tbl.to_csv(out_base + '.csv', index=False)

    ymax = max(4, int(tbl['n_mismatch'].max()))
    fig, axes, shape = _panel_grid(len(arms), ncols,
                                   0.42 * len(viruses) + 1.1, 3.1)
    scale = 5.0

    for i, (ax, arm) in enumerate(zip(axes, arms)):
        ax.set_axisbelow(True)
        ax.grid(axis='y', color=GRID, linewidth=0.8)
        for xi, virus in enumerate(viruses):
            sub = tbl[(tbl['arm'] == arm) & (tbl['virus'] == virus)]
            if sub.empty:
                continue
            ax.scatter(np.full(len(sub), xi), sub['n_mismatch'],
                       s=sub['pct'] * scale, color=SERIES[0], alpha=0.85,
                       edgecolors=SURFACE, linewidths=1.6, zorder=3)
        ax.set_xticks(range(len(viruses)))
        if _is_bottom(i, len(arms), shape):
            ax.set_xticklabels(viruses, rotation=55, ha='right', fontsize=8)
        else:
            ax.set_xticklabels([])
        ax.set_xlim(-0.7, len(viruses) - 0.3)
        # the 100%-of-pairs bubble at zero is wide; give it room
        ax.set_ylim(-0.045 * ymax - 1.0, ymax + 0.8)
        ax.set_title(arm, color=INK, fontsize=10.5, pad=6)
        _frame(ax)

    for i, ax in enumerate(axes):
        if i % shape[1] == 0:
            ax.set_ylabel('nt mismatches per primer', color=INK_2, fontsize=9.5)
    fig.suptitle('Mismatches between primers and alignment sequences',
                 color=INK, fontsize=13, x=0.012, ha='left', y=0.995)

    handles = [plt.scatter([], [], s=p * scale, color=SERIES[0], alpha=0.85,
                           edgecolors=SURFACE, linewidths=1.6, label=str(p))
               for p in (1, 10, 100)]
    leg = fig.legend(handles=handles, title='% of pairs', loc='center left',
                     bbox_to_anchor=(0.995, 0.5), frameon=False,
                     labelspacing=2.2, borderpad=0.8, handletextpad=1.6,
                     fontsize=9)
    leg.get_title().set_color(INK_2)
    leg.get_title().set_fontsize(9)
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_base + '.png', dpi=200, facecolor=SURFACE, bbox_inches='tight')
    fig.savefig(out_base + '.pdf', facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def fig4b(positions, arms, viruses, out_base, penalised=5, min_support=0.2, ncols=3):
    '''One panel per tool; within a panel one line per virus, mismatch rate by distance
    from the primer's 3' end.
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Manuscript figure style: 131 mm wide, 7-8 pt sans-serif, TrueType fonts.
    rc = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans',
                            'Nimbus Sans', 'DejaVu Sans'],
        'font.size': 8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'axes.unicode_minus': False,
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.5,
        'axes.labelcolor': 'black',
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.color': 'black',
        'ytick.color': 'black',
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.fontsize': 7,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'lines.solid_capstyle': 'round',
        'lines.solid_joinstyle': 'round',
    }
    MM = 1 / 25.4
    BAND = '#e6e6e6'

    positions.to_csv(out_base + '.csv', index=False)

    # Drop positions with too little support.
    keep = []
    for (arm, virus), sub in positions.groupby(['arm', 'virus']):
        keep.append(sub[sub['n_scored'] >= min_support * sub['n_scored'].max()])
    plotted = pd.concat(keep, ignore_index=True) if keep else positions

    xmax = int(plotted['dist_from_3prime'].max())
    ymax = max(1.0, float(plotted['pct_mismatch'].max()))

    with plt.rc_context(rc):
        nrows = int(np.ceil(len(arms) / float(ncols)))
        fig, grid = plt.subplots(nrows, ncols,
                                 figsize=(131 * MM, (42 * nrows + 4) * MM),
                                 sharey=True, squeeze=False)
        flat = [ax for row in grid for ax in row]
        for ax in flat[len(arms):]:
            ax.set_visible(False)
        # A free grid slot holds the legend; otherwise it goes above the panels.
        legend_ax = flat[len(arms)] if len(flat) > len(arms) else None
        axes, shape = flat[:len(arms)], (nrows, ncols)
        # Unknown (ad-hoc --msa) viruses fall back to SERIES, skipping hues
        # already taken so no two lines share a colour.
        spare = iter([c for c in SERIES if c not in VIRUS_COLOURS.values()])
        colours = {v: VIRUS_COLOURS.get(v) or next(spare, MUTED) for v in viruses}

        for i, (ax, arm) in enumerate(zip(axes, arms)):
            ax.axvspan(-0.5, penalised - 0.5, color=BAND, zorder=0, lw=0)
            for virus in viruses:
                sub = plotted[(plotted['arm'] == arm) & (plotted['virus'] == virus)]
                if sub.empty:
                    continue
                sub = sub.sort_values('dist_from_3prime')
                ax.plot(sub['dist_from_3prime'], sub['pct_mismatch'],
                        color=colours[virus], linewidth=0.9,
                        zorder=3, label=virus)
            ax.set_xlim(-0.8, xmax + 0.5)
            ax.set_ylim(0, ymax * 1.06)
            ax.set_title(arm, pad=3)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            if _is_bottom(i, len(arms), shape) or (legend_ax is not None and
                                                  i + shape[1] == len(arms)):
                ax.set_xlabel("Distance from 3′ end (nt)", labelpad=2)
            if i % shape[1] == 0:
                ax.set_ylabel('Mismatch (%)', labelpad=2)
        axes[0].text(-0.3, ymax * 1.05, "3′ end",
                     ha='left', va='top', fontsize=6.5)

        # Ten series cross constantly, so identity goes in one shared legend
        # above the panels, kept inside the 180 mm width.
        handles = [plt.Line2D([], [], color=colours[v], linewidth=1.2, label=v)
                   for v in viruses]
        if legend_ax is not None:
            legend_ax.set_visible(True)
            legend_ax.axis('off')
            legend_ax.legend(handles=handles, loc='center', ncol=2,
                             frameon=False, handlelength=1.4, handletextpad=0.5,
                             columnspacing=1.0, labelspacing=0.45, borderaxespad=0)
            fig.tight_layout(w_pad=0.8, h_pad=0.8)
        else:
            fig.legend(handles=handles, loc='upper center',
                       bbox_to_anchor=(0.5, 1.0), ncol=min(len(viruses), 6),
                       frameon=False, handlelength=1.4, handletextpad=0.5,
                       columnspacing=1.2, borderaxespad=0)
            fig.tight_layout(rect=(0, 0, 1, 1 - 9 * MM / fig.get_figheight()),
                             w_pad=0.8, h_pad=0.8)
        # 600 dpi is the usual raster floor for a line figure in print.
        fig.savefig(out_base + '.png', dpi=600)
        fig.savefig(out_base + '.pdf')
        plt.close(fig)


def run_one_target(virus, msa_path, primer_specs, args):
    '''Place and score every arm's primers against one alignment.'''
    print('\n### %s -- %s' % (virus, msa_path))
    msa = MSA(msa_path, args.threads)
    print('  %d sequences, %d columns, %d nt gapless consensus'
          % (msa.row, msa.col, len(msa.consensus)))

    tables = {}
    order = []
    for label, path in primer_specs:
        for lbl, rows in load_primers(label, path).items():
            tables[lbl] = rows
            if lbl not in order:
                order.append(lbl)

    pairs, positions, summary = [], [], []
    for label in order:
        if args.arms and label not in args.arms:
            continue
        rows = tables[label]
        p, q, skipped = analyse(virus, label, rows, msa,
                                args.min_score_frac, args.ambiguous)
        if p.empty:
            print('  %-14s no primer could be placed' % label)
            continue
        pairs.append(p)
        positions.append(q)
        deg = p.drop_duplicates('primer_name')['n_permutations']
        summary.append({
            'virus': virus,
            'arm': label,
            'n_primers_placed': p['primer_name'].nunique(),
            'n_primers_input': len(rows),
            'n_pairs': len(p),
            'mean_mismatches_per_primer': p['n_mismatch'].mean(),
            'pct_pairs_perfect': 100.0 * (p['n_mismatch'] == 0).mean(),
            'pct_pairs_ge2': 100.0 * (p['n_mismatch'] >= 2).mean(),
            'mean_permutations': deg.mean(),
            'max_permutations': int(deg.max()),
            'pct_primers_degenerate': 100.0 * (deg > 1).mean(),
            'mean_primer_len': p.drop_duplicates('primer_name')['primer_len'].mean(),
        })
        print('  %-14s placed %3d/%3d  mean mismatches %.3f  perfect pairs %5.1f%%%s'
              % (label, summary[-1]['n_primers_placed'], len(rows),
                 summary[-1]['mean_mismatches_per_primer'],
                 summary[-1]['pct_pairs_perfect'],
                 '  (%d skipped)' % len(skipped) if skipped else ''))
    return pairs, positions, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--targets', nargs='*', default=None,
                    help='target names to run; default all of: %s'
                         % ', '.join(TARGET_ORDER))
    ap.add_argument('--msa', default=None,
                    help='single ad-hoc alignment instead of --targets')
    ap.add_argument('--virus', default=None,
                    help='label for --msa [basename]')
    ap.add_argument('--primers', action='append', default=None, metavar='LABEL=PATH',
                    help='with --msa: repeatable primer file; LABEL= may be '
                         'omitted for a collected designed_primers.csv, whose '
                         'tool column is used instead')
    ap.add_argument('--arms', nargs='*', default=None,
                    help='restrict to these tool arms')
    ap.add_argument('--out', required=True)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--ncols', type=int, default=3,
                    help='panels per row in Fig 4a [3]')
    ap.add_argument('--ncols-4b', type=int, default=None,
                    help='panels per row in Fig 4b [all tools in one row]')
    ap.add_argument('--replot-from', default=None, metavar='DIR',
                    help='re-render the figures from a previous run\'s '
                         'mismatch_pairs.csv and fig4b CSV instead of placing '
                         'primers again (the placement pass is the slow half)')
    ap.add_argument('--min-score-frac', type=float, default=0.6,
                    help='reject a placement scoring below this times primer length [0.6]')
    ap.add_argument('--min-support', type=float, default=0.2,
                    help="Fig 4b drops positions scored on fewer than this "
                         "fraction of a cell's peak pair count [0.2]")
    ap.add_argument('--ambiguous', choices=AMBIGUOUS_MODES, default='resolve',
                    help="how an ambiguity code in a TARGET sequence is "
                         "treated. 'resolve' (default) counts a mismatch only "
                         "where no resolution of the target base is compatible "
                         "with any permutation of the primer, so W matches a T "
                         "primer and mismatches a G one; 'drop' excludes the "
                         "(primer, sequence) pair; 'match' never counts one as "
                         "a mismatch; 'mismatch' always does. Ambiguity in the "
                         "primer is always resolved, in every mode.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.replot_from:
        pairs = pd.read_csv(os.path.join(args.replot_from, 'mismatch_pairs.csv'))
        positions = pd.read_csv(os.path.join(
            args.replot_from, 'fig4b_mismatch_by_position.csv'))
        summary_df = pd.read_csv(os.path.join(args.replot_from, 'summary.csv'))
        _render(pairs, positions, summary_df, args)
        return

    jobs = []
    if args.msa:
        if not args.primers:
            raise SystemExit('--msa needs at least one --primers')
        specs = []
        for spec in args.primers:
            label, _, path = spec.partition('=')
            if not path:
                label, path = '', label
            specs.append((label, path))
        jobs.append((args.virus or os.path.basename(args.msa).split('.')[0],
                     args.msa, specs))
    else:
        names = args.targets or TARGET_ORDER
        unknown = [n for n in names if n not in TARGETS]
        if unknown:
            raise SystemExit('unknown target(s): %s; known: %s'
                             % (', '.join(unknown), ', '.join(TARGET_ORDER)))
        for name in names:
            msa_path, out_dir = TARGETS[name]
            collected = os.path.join(out_dir, 'designed_primers.csv')
            if not os.path.isfile(collected):
                print('skipping %s: no %s' % (name, collected))
                continue
            jobs.append((name, msa_path, [('', collected)]))

    all_pairs, all_positions, summary = [], [], []
    for virus, msa_path, specs in jobs:
        p, q, s = run_one_target(virus, msa_path, specs, args)
        all_pairs += p
        all_positions += q
        summary += s

    if not all_pairs:
        raise SystemExit('nothing could be placed')

    _render(pd.concat(all_pairs, ignore_index=True),
            pd.concat(all_positions, ignore_index=True),
            pd.DataFrame(summary), args)


def _render(pairs, positions, summary_df, args):
    # --arms and --targets are applied here as well as at placement time.
    if args.arms:
        keep = set(args.arms)
        pairs = pairs[pairs['arm'].isin(keep)]
        positions = positions[positions['arm'].isin(keep)]
        summary_df = summary_df[summary_df['arm'].isin(keep)]
    if args.targets:
        keep = set(args.targets)
        pairs = pairs[pairs['virus'].isin(keep)]
        positions = positions[positions['virus'].isin(keep)]
        summary_df = summary_df[summary_df['virus'].isin(keep)]
    if pairs.empty:
        raise SystemExit('no rows left after --arms/--targets')

    viruses = [v for v in TARGET_ORDER if v in set(pairs['virus'])]
    viruses += [v for v in dict.fromkeys(pairs['virus']) if v not in viruses]
    arms = [a for a in ARM_ORDER if a in set(pairs['arm'])]
    arms += [a for a in dict.fromkeys(pairs['arm']) if a not in arms]
    if not args.arms:
        # A reference scheme that exists for one target only (ARTIC, SARS-CoV-2)
        # would otherwise take a panel that is blank in five columns out of six.
        covered = pairs.groupby('arm')['virus'].nunique()
        dropped = [a for a in arms if covered[a] * 2 < len(viruses)]
        if dropped:
            print('\nomitting from the figures (fewer than half the targets): %s'
                  '\n  -- still in summary.csv and the matrix; --arms overrides'
                  % ', '.join(dropped))
        arms = [a for a in arms if a not in dropped]

    pairs.to_csv(os.path.join(args.out, 'mismatch_pairs.csv'), index=False)
    summary_df.to_csv(os.path.join(args.out, 'summary.csv'), index=False)

    fig4a(pairs, arms, viruses, os.path.join(args.out, 'fig4a_mismatch_counts'),
          ncols=args.ncols)
    fig4b(positions, arms, viruses,
          os.path.join(args.out, 'fig4b_mismatch_by_position'),
          min_support=args.min_support, ncols=args.ncols_4b or len(arms))

    # Table 2's last row, as a virus x tool matrix
    matrix = (summary_df.pivot(index='arm', columns='virus',
                               values='mean_mismatches_per_primer')
              .reindex(index=arms, columns=viruses))
    matrix.to_csv(os.path.join(args.out, 'mean_mismatches_matrix.csv'))
    print('\n--- mean mismatches per primer (Table 2, last row) ---')
    print(matrix.round(3).to_string())
    print('\nwrote %s' % args.out)


if __name__ == '__main__':
    main()
