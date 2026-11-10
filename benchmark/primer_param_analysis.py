#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Primer parameter panels (alignment entropy, degeneracy, Tm, hairpin, homodimer, GC).

    python primer_param_analysis.py --msa targets/hev/hev_stratified_aligned.fasta
    \\
        --primers Olivia-deg=output/HEV_deg/Olivia/amp400/design.csv --out
        output/HEV_deg/params

With --entropy-msa LABEL=PATH for other targets it also draws the entropy grid (paper
Fig. 2).
'''

import argparse
import itertools
import math
import multiprocessing
import os
import sys

import numpy as np
import pandas as pd
from Bio import SeqIO

import config

sys.path.insert(0, config.SRC_DIR)
sys.path.insert(0, config.PACKAGE_DIR)
from msa_tools import MSA
from primer_gate import PCR_CONDITIONS, VARVAMP_GATE

# Shared with mismatch_analysis so both figures read primers the same way.
from mismatch_analysis import (RESOLVE, COMPLEMENT, load_primers, locate,
                               SURFACE, INK, INK_2, MUTED, GRID, SERIES,
                               _frame as frame)


# varVAMP's optima, the two it has that the gate does not carry (the gate only
# needs limits): default_config.PRIMER_TMP[2] and PRIMER_GC_RANGE[2].
TM_OPT = 60.0
GC_OPT = 50.0

# Series colours.
C_OPT = SERIES[1]     # opt (most common nt)
C_PERM = SERIES[0]    # mean (all permutations)

# Five colour-blind-safe hues for the entropy lines.
ENTROPY_SERIES = ['#2a78d6', '#eda100', '#e87ba4', '#008300', '#4a3aa7']

# Targets 6-10 reuse the hues with a dashed line.
ENTROPY_STYLES = ['-', '--']


# Primer parameters

def expansions(seq, cap):
    '''(sampled permutations, true permutation count).'''
    options = [sorted(RESOLVE.get(b, set('ACGT'))) for b in seq]
    n_true = 1
    for o in options:
        n_true *= len(o)
    it = itertools.product(*options)
    return [''.join(p) for p in itertools.islice(it, cap)], n_true


def gc_percent(seq):
    return 100.0 * sum(1 for b in seq if b in 'GC') / max(len(seq), 1)


def _params(seq):
    '''The four panel quantities for one concrete ACGT sequence, plus the two
    structure_found flags.
    '''
    import primer3
    hp = primer3.calc_hairpin(seq, **PCR_CONDITIONS)
    hd = primer3.calc_homodimer(seq, **PCR_CONDITIONS)
    return (primer3.calc_tm(seq, **PCR_CONDITIONS), hp.tm, hd.tm, gc_percent(seq),
            float(hp.structure_found), float(hd.structure_found))


def _worker(job):
    '''One primer -> one row.'''
    (label, name, direction, seq, opt_seq, opt_source, cap, expand) = job

    if expand:
        perms, n_true = expansions(seq, cap)
    else:
        perms, n_true = [seq], 1
    vals = np.array([_params(p) for p in perms], dtype=np.float64)
    mean = vals.mean(axis=0)

    opt = _params(opt_seq) if opt_seq is not None else (np.nan,) * 6
    n_amb = sum(1 for b in seq if b not in 'ACGT')
    return {
        'arm': label, 'primer_name': name, 'direction': direction,
        'primer': seq, 'opt_primer': opt_seq or '', 'opt_source': opt_source,
        'primer_len': len(seq), 'n_ambiguous': n_amb,
        'n_permutations': n_true, 'n_permutations_scored': len(perms),
        'tm_opt': opt[0], 'hairpin_opt': opt[1],
        'homodimer_opt': opt[2], 'gc_opt': opt[3],
        'tm_mean': mean[0], 'hairpin_mean': mean[1],
        'homodimer_mean': mean[2], 'gc_mean': mean[3],
        # fraction of permutations primer3 found a fold for; 0 means every
        # permutation's tm above is the "no structure" sentinel
        'hairpin_found_opt': opt[4], 'homodimer_found_opt': opt[5],
        'hairpin_found_frac': mean[4], 'homodimer_found_frac': mean[5],
    }


def revcomp(seq):
    return ''.join(COMPLEMENT.get(b, 'N') for b in reversed(seq))


def build_jobs(label, primers, msa, cap, expand, min_score_frac):
    '''Placement happens here, serially, because it needs the alignment.'''
    jobs, skipped = [], []
    n_cons = len(msa.consensus) if msa is not None else 0

    for name, direction, seq in primers:
        seq = seq.upper().replace('U', 'T')
        if len(seq) < 2 or not set(seq) <= set(RESOLVE):
            skipped.append((name, 'unreadable sequence'))
            continue

        degenerate = any(b not in 'ACGT' for b in seq)
        if not degenerate:
            jobs.append((label, name, direction, seq, seq, 'designed', cap, expand))
            continue
        if msa is None:
            jobs.append((label, name, direction, seq, None, 'unplaced', cap, expand))
            continue

        start, stop, strand, score = locate(seq, msa.consensus)
        if direction.startswith('rev'):
            strand = '-'
        elif direction.startswith('for'):
            strand = '+'
        if score < min_score_frac * len(seq):
            skipped.append((name, 'weak alignment (score %.1f over %d nt); '
                                  'opt point dropped' % (score, len(seq))))
            jobs.append((label, name, direction, seq, None, 'unplaced', cap, expand))
            continue
        if start < 0 or stop > n_cons or stop - start != len(seq):
            skipped.append((name, 'indel or off-end placement (%d-%d for %d nt); '
                                  'opt point dropped' % (start, stop, len(seq))))
            jobs.append((label, name, direction, seq, None, 'unplaced', cap, expand))
            continue

        opt_seq = msa.consensus[start:stop]
        if strand == '-':
            opt_seq = revcomp(opt_seq)
        # The majority consensus is ACGT by construction (argmax over 5 bases),
        # but a gap can win a column, and '-' is not a sequence primer3 reads.
        if not set(opt_seq) <= set('ACGT'):
            skipped.append((name, 'majority consensus has a gap under the primer; '
                                  'opt point dropped'))
            opt_seq = None
        jobs.append((label, name, direction, seq, opt_seq,
                     'consensus' if opt_seq else 'unplaced', cap, expand))

    return jobs, skipped


# Alignment statistics (panel a, and the GC reference on panel f)

def _pretty(path):
    '''Filename to series label: the suffixes every target in this benchmark
    shares carry no information once the alignments are side by side.'''
    stem = os.path.splitext(os.path.basename(path))[0]
    for suffix in ('_stratified_aligned', '_reference35_aligned', '_aligned', '_aln'):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def read_matrix(path):
    '''(n_sequences, n_columns) uppercase character matrix.'''
    rows = [str(r.seq).upper() for r in SeqIO.parse(path, 'fasta')]
    if not rows:
        raise SystemExit('no sequences in %s' % path)
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise SystemExit('%s is not aligned (ragged rows)' % path)
    return np.array([list(r) for r in rows], dtype='<U1')


def column_entropy(matrix, ignore_ambiguous=True):
    '''Normalised Shannon entropy per column (base 4, in [0, 1]); optionally ignores
    ambiguity codes.
    '''
    keep = list('ACGT') if ignore_ambiguous else None
    ent = np.zeros(matrix.shape[1], dtype=np.float64)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        col = col[np.isin(col, keep)] if keep else col[col != '-']
        n = col.size
        if n <= 1:
            continue
        _, counts = np.unique(col, return_counts=True)
        if np.count_nonzero(counts) <= 1:
            continue
        p = counts / n
        ent[j] = -np.sum(p * np.log(p) / math.log(4))
    return ent


def entropy_profile(path, window_frac=0.01, ignore_ambiguous=True):
    matrix = read_matrix(path)
    ent = column_entropy(matrix, ignore_ambiguous)
    window = max(1, int(round(window_frac * len(ent))))
    smooth = pd.Series(ent).rolling(window, center=True, min_periods=1).mean().to_numpy()
    pct = 100.0 * np.arange(len(ent)) / max(len(ent) - 1, 1)

    acgt = np.isin(matrix, list('ACGT'))
    gc = np.isin(matrix, list('GC'))
    gc_pct = 100.0 * gc.sum() / max(acgt.sum(), 1)
    return pct, ent, smooth, gc_pct, matrix.shape


# Panels

def _resample(x, y, n_out, how='mean'):
    """Bin a profile down to n_out points."""
    if len(y) <= n_out:
        return x, y
    edges = np.linspace(0, len(y), n_out + 1).astype(int)
    out = np.empty(n_out, dtype=np.float64)
    xs = np.empty(n_out, dtype=np.float64)
    for i in range(n_out):
        a, b = edges[i], max(edges[i + 1], edges[i] + 1)
        seg = y[a:b]
        out[i] = seg.max() if how == 'max' else seg.mean()
        xs[i] = x[a:b].mean()
    return xs, out


def _strip_x(n, xi, rng, width=0.26):
    '''Deterministic jitter, so a rerun redraws the same figure.'''
    if n == 0:
        return np.array([])
    return xi + rng.uniform(-width, width, size=n)


def panel_a(ax, profiles, ignore_ambiguous=True):
    '''Entropy against position.'''
    single = len(profiles) == 1
    n_hue = len(ENTROPY_SERIES)
    if len(profiles) > n_hue * len(ENTROPY_STYLES):
        raise SystemExit('panel a cannot encode %d targets; use the grid figure'
                         % len(profiles))
    for i, (label, pct, ent, smooth, _) in enumerate(profiles):
        colour = SERIES[0] if single else ENTROPY_SERIES[i % n_hue]
        style = ENTROPY_STYLES[i // n_hue]
        if single:
            fx, fy = _resample(pct, ent, 1200, how='max')
            ax.fill_between(fx, fy, color=GRID, linewidth=0, zorder=1,
                            label='per column')
        lx, ly = _resample(pct, smooth, 1200)
        ax.plot(lx, ly, color=colour, linewidth=1.6, linestyle=style,
                zorder=3, label=label)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(1e-3, max(float(np.nanmax(p[3])) for p in profiles) * 1.15))
    ax.set_xlabel('% multiple sequence alignment', color=INK_2, fontsize=10)
    ax.set_ylabel("normalized Shannon's entropy", color=INK_2, fontsize=10)
    # The two filtering rules give very different curves on a masked
    # alignment, so the figure has to say which one it is holding.
    ax.set_title('a   Alignment entropy (rolling average over 1%% of columns, %s)'
                 % ('ACGT only' if ignore_ambiguous else "varVAMP rule, 'N' counted"),
                 color=INK, fontsize=11, loc='left', pad=10)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    frame(ax)
    leg = ax.legend(frameon=False, fontsize=9, loc='upper right',
                    ncol=2 if len(profiles) > 5 else 1, columnspacing=1.4)
    for t in leg.get_texts():
        t.set_color(INK_2)


def _grid_shape(n):
    '''Grid shape for n panels with the fewest empty slots.'''
    best = min(range(3, 6), key=lambda c: ((-n) % c, -c))
    return best


def draw_entropy_grid(profiles, out_base, ignore_ambiguous=True, ncol=None,
                      raw=False):
    """Panel a as small multiples: one mini-panel per target, shared axes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    order = sorted(profiles, key=lambda p: float(np.nanmean(p[2])))
    ncol = ncol or _grid_shape(len(order))
    nrow = int(np.ceil(len(order) / ncol))
    ymax = max(float(np.nanmax(p[3])) for p in order) * 1.12

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
        'mathtext.default': 'regular',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.5,
        'axes.labelcolor': 'black',
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
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'lines.solid_capstyle': 'round',
        'lines.solid_joinstyle': 'round',
    }
    MM = 1 / 25.4
    print('  entropy grid: rolling average over 1%% of columns, %s'
          % ('ACGT only' if ignore_ambiguous else "varVAMP rule, 'N' counted"))

    with plt.rc_context(rc):
        # constrained_layout, as in the other manuscript figures.
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(131 * MM, 38 * nrow * MM),
                                 sharex=True, sharey=True, squeeze=False,
                                 layout='constrained')
        fig.get_layout_engine().set(w_pad=0.5 * MM, h_pad=1 * MM,
                                    wspace=0.01, hspace=0.02)

        for k, ax in enumerate(axes.ravel()):
            if k >= len(order):
                ax.axis('off')
                continue
            label, pct, ent, smooth, _ = order[k]
            # The per-column layer is off here by default.
            if raw:
                fx, fy = _resample(pct, ent, 600, how='max')
                ax.fill_between(fx, fy, color='#e6e6e6', linewidth=0, zorder=1)
            lx, ly = _resample(pct, smooth, 600)
            ax.plot(lx, ly, color=SERIES[0], linewidth=0.8, zorder=3)
            ax.set_ylim(0, ymax)
            ax.set_xlim(0, 100)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MultipleLocator(50))
            # The panel name and its mean, so the ranking is readable without measuring
            # curves.
            ax.set_title(label, loc='left', fontsize=8, fontweight='bold', pad=12)
            # Stacked under the name: at 131 mm over four columns the long names
            # (SARS-CoV-2) run into it on a shared line.
            ax.text(0, 1.02, r'$\bar{H}$ = %.3f' % float(np.nanmean(ent)),
                    transform=ax.transAxes, ha='left', va='bottom',
                    fontsize=7, color='#4d4d4d')

        for c in range(ncol):
            col = [axes[r][c] for r in range(nrow) if r * ncol + c < len(order)]
            if col:
                col[-1].set_xlabel('Position in MSA (%)', labelpad=2)
                col[-1].tick_params(labelbottom=True)
        for row in axes:
            row[0].set_ylabel('Normalized entropy', labelpad=2)

        fig.savefig(out_base + '.png', dpi=600)
        fig.savefig(out_base + '.pdf')
        plt.close(fig)


def panel_b(ax, df, arms, rng):
    '''Degeneracy per primer.'''
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    vmax = 1
    for xi, arm in enumerate(arms):
        sub = df[df['arm'] == arm]
        y = sub['n_permutations'].to_numpy(dtype=float)
        vmax = max(vmax, int(y.max()) if y.size else 1)
        ax.scatter(_strip_x(y.size, xi, rng), y, s=26, color=C_PERM, alpha=0.7,
                   edgecolors=SURFACE, linewidths=0.8, zorder=3)
        if y.size:
            ax.plot([xi - 0.34, xi + 0.34], [y.mean()] * 2, color=INK,
                    linewidth=2.0, zorder=4, solid_capstyle='butt')
            ax.annotate('%d' % sub['n_ambiguous'].max(), (xi, 1.0),
                        xycoords=('data', 'axes fraction'),
                        textcoords='offset points', xytext=(0, 6),
                        ha='center', color=MUTED, fontsize=9)

    if vmax > 100:
        # Without an -a cap one primer can sit three orders of magnitude above
        # the rest and flatten every other column into the axis.
        ax.set_yscale('symlog', linthresh=10)
    ax.set_ylim(0, vmax * 1.25)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=20, ha='right')
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.set_ylabel('n permutations per primer', color=INK_2, fontsize=10)
    ax.set_title('b   Degeneracy\n     (max n ambiguous nt above each column)',
                 color=INK, fontsize=11, loc='left', pad=26)
    frame(ax)


SERIES_LABELS = ('opt (most common nt)', 'mean (all permutations)')


def clip_floor(df, cols, percentile):
    """Lower bound for a panel's y-axis."""
    if not percentile:
        return None
    pooled = np.concatenate([df[c].to_numpy(dtype=float) for c in cols])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return None
    lo = float(np.percentile(pooled, percentile))
    return lo if (pooled < lo).any() else None


def panel_param(ax, df, arms, rng, opt_col, mean_col, title, ylabel,
                limits=(), legend=False, clip_lo=None):
    """One of panels c-f, with reference lines at `limits`."""
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=GRID, linewidth=0.8)

    for style_label, value, ls in limits:
        ax.axhline(value, color=INK_2, linewidth=1.0, linestyle=ls, zorder=2,
                   alpha=0.75)
        ax.annotate(style_label, (1.0, value), xycoords=('axes fraction', 'data'),
                    textcoords='offset points', xytext=(4, 0), va='center',
                    color=MUTED, fontsize=8, annotation_clip=False)

    n_clipped = 0
    for xi, arm in enumerate(arms):
        sub = df[df['arm'] == arm]
        for col, colour, off, lab in ((opt_col, C_OPT, -0.17, SERIES_LABELS[0]),
                                      (mean_col, C_PERM, 0.17, SERIES_LABELS[1])):
            y = sub[col].to_numpy(dtype=float)
            y = y[np.isfinite(y)]
            x = _strip_x(y.size, xi + off, rng, width=0.13)
            below = np.zeros(y.shape, dtype=bool) if clip_lo is None else y < clip_lo
            n_clipped += int(below.sum())
            ax.scatter(x[~below], y[~below], s=22, color=colour, alpha=0.75,
                       edgecolors=SURFACE, linewidths=0.8, zorder=3,
                       label=lab if (xi == 0 and legend) else None)
            if below.any():
                ax.scatter(x[below], np.full(int(below.sum()), clip_lo), s=26,
                           marker='v', facecolors='none', edgecolors=colour,
                           linewidths=1.2, zorder=4)

    if clip_lo is not None:
        span = ax.get_ylim()[1] - clip_lo
        ax.set_ylim(clip_lo - 0.06 * span, ax.get_ylim()[1])
        ax.annotate('%d below axis' % n_clipped, (0.0, clip_lo),
                    xycoords=('axes fraction', 'data'),
                    textcoords='offset points', xytext=(2, -13),
                    color=MUTED, fontsize=8)

    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=20, ha='right')
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11, loc='left', pad=10)
    frame(ax)
    if legend:
        leg = ax.legend(frameon=False, fontsize=9, loc='best', handletextpad=0.3)
        for t in leg.get_texts():
            t.set_color(INK_2)


def panel_specs(design_gc, msa_gc):
    '''(key, title, ylabel, limits) for c-f, in the paper's order.'''
    tm_lo, tm_hi = VARVAMP_GATE['tm']
    gc_lo, gc_hi = VARVAMP_GATE['gc']
    gc_limits = [('min', gc_lo, ':'), ('max', gc_hi, ':'), ('target', GC_OPT, '--')]
    if msa_gc is not None:
        gc_limits.append(('MSA GC %.0f%%' % msa_gc, msa_gc, '-.'))
    if design_gc is not None:
        gc_limits += [('design min', design_gc[0], (0, (1, 3))),
                      ('design max', design_gc[1], (0, (1, 3)))]
    return [
        ('c', 'tm_opt', 'tm_mean', 'c   Melting temperature', '°C melting temp',
         [('min', tm_lo, ':'), ('max', tm_hi, ':'), ('target', TM_OPT, '--')]),
        ('d', 'hairpin_opt', 'hairpin_mean', 'd   Hairpin', '°C hairpin temp',
         [('max', VARVAMP_GATE['hairpin_tm_max'], '--')]),
        ('e', 'homodimer_opt', 'homodimer_mean', 'e   Homo-dimer', '°C homo-dimer temp',
         [('max', VARVAMP_GATE['homodimer_tm_max'], '--')]),
        ('f', 'gc_opt', 'gc_mean', 'f   GC content', '% GC content', gc_limits),
    ]


def draw(df, arms, profiles, out_dir, design_gc, msa_gc, seed, separate,
         clip_percentile, ignore_ambiguous=True, grid_raw=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    specs = panel_specs(design_gc, msa_gc)
    # Only the two structure panels get a floor; Tm and GC are bounded
    # quantities with no sentinel value to escape to.
    floors = {spec[1]: (clip_floor(df, [spec[1], spec[2]], clip_percentile)
                        if spec[0] in ('d', 'e') else None)
              for spec in specs}

    def save(fig, base):
        fig.savefig(base + '.png', dpi=200, facecolor=SURFACE, bbox_inches='tight')
        fig.savefig(base + '.pdf', facecolor=SURFACE, bbox_inches='tight')
        plt.close(fig)

    # Combined, the paper's layout: a spanning three quarters of the top row
    # with b beside it, c-f across the bottom.
    width = max(13.0, 2.4 * len(arms) + 8.0)
    fig = plt.figure(figsize=(width, 9.2))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.15], hspace=0.62, wspace=0.42)
    ax_a = fig.add_subplot(gs[0, :3])
    ax_b = fig.add_subplot(gs[0, 3])
    axes_cf = [fig.add_subplot(gs[1, i]) for i in range(4)]
    for ax in [ax_a, ax_b] + axes_cf:
        ax.set_facecolor(SURFACE)

    panel_a(ax_a, profiles, ignore_ambiguous)
    panel_b(ax_b, df, arms, np.random.default_rng(seed))
    for ax, (_, opt_c, mean_c, title, ylabel, limits) in zip(axes_cf, specs):
        panel_param(ax, df, arms, np.random.default_rng(seed), opt_c, mean_c,
                    title, ylabel, limits, clip_lo=floors[opt_c])

    # One legend for the bottom row.
    handles = [plt.Line2D([], [], marker='o', linestyle='none', markersize=7,
                          markerfacecolor=c, markeredgecolor=SURFACE, label=l)
               for c, l in ((C_OPT, SERIES_LABELS[0]), (C_PERM, SERIES_LABELS[1]))]
    leg = fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False,
                     bbox_to_anchor=(0.5, 0.52), fontsize=10, handletextpad=0.4,
                     columnspacing=2.0)
    for t in leg.get_texts():
        t.set_color(INK_2)

    save(fig, os.path.join(out_dir, 'fig3_primer_parameters'))

    if not separate:
        return
    # Panels also go out one per file: a paper places them individually and
    # cropping a composite loses the vector text.
    if len(profiles) > 1:
        draw_entropy_grid(profiles, os.path.join(out_dir,
                                                 'fig3a_alignment_entropy_grid'),
                          ignore_ambiguous, raw=grid_raw)

    names = ['fig3a_alignment_entropy', 'fig3b_degeneracy'] + \
            ['fig3%s_%s' % (k, c.replace('_opt', '')) for k, c, _, _, _, _ in specs]
    for idx, name in enumerate(names):
        fig = plt.figure(figsize=(8.0, 4.4) if idx == 0 else
                         (max(4.6, 1.3 * len(arms) + 2.8), 4.6))
        fig.patch.set_facecolor(SURFACE)
        ax = fig.add_subplot(111)
        ax.set_facecolor(SURFACE)
        if idx == 0:
            panel_a(ax, profiles, ignore_ambiguous)
        elif idx == 1:
            panel_b(ax, df, arms, np.random.default_rng(seed))
        else:
            _, opt_c, mean_c, title, ylabel, limits = specs[idx - 2]
            panel_param(ax, df, arms, np.random.default_rng(seed), opt_c, mean_c,
                        title, ylabel, limits, legend=True,
                        clip_lo=floors[opt_c])
        fig.tight_layout()
        save(fig, os.path.join(out_dir, name))


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--msa', required=True,
                    help='the alignment the primers were designed on: panel a, '
                         'the GC reference on f, and the majority consensus the '
                         'opt series is read from')
    ap.add_argument('--primers', action='append', required=True, metavar='LABEL=PATH',
                    help='repeatable; LABEL= may be omitted for a collected '
                         'designed_primers.csv, whose tool column is used instead')
    ap.add_argument('--out', required=True)
    ap.add_argument('--deg', default='auto', type=str.lower,
                    choices=('auto', 'true', 'false'),
                    help="expand degenerate primers into their permutations: "
                         "'true' always, 'false' never (every primer read "
                         "literally, so c-f carry one series), 'auto' when any "
                         "primer carries an IUPAC code [auto]")
    ap.add_argument('--msa-label', default=None,
                    help='series name for --msa in panel a [filename stem, '
                         'with this benchmark\'s shared suffixes stripped]')
    ap.add_argument('--entropy-msa', action='append', default=[], metavar='LABEL=PATH',
                    help='repeatable; extra alignments drawn in panel a only, '
                         'for the multi-target version of the paper figure')
    ap.add_argument('--max-permutations', type=int, default=1024,
                    help='permutations averaged per primer in c-f; the true '
                         'count is still what b plots [1024]')
    ap.add_argument('--design-gc', default='20,75',
                    help='design-time GC window drawn on panel f, as min,max '
                         'percent -- Olivar\'s min_GC/max_GC from '
                         'design_olivia*.py. "none" omits it [20,75]')
    ap.add_argument('--min-score-frac', type=float, default=0.6,
                    help='reject a placement scoring below this times primer '
                         'length; its opt point is dropped [0.6]')
    ap.add_argument('--entropy-ignore-ambiguous',
                    action=argparse.BooleanOptionalAction, default=True,
                    help="drop ambiguity codes from the entropy calculation "
                         "alongside gaps, so an 'N' reads as missing data "
                         "rather than as a fifth nucleotide. --no-... restores "
                         "varVAMP's exact rule; see column_entropy for what "
                         "that costs on a masked alignment [on]")
    ap.add_argument('--grid-raw', action='store_true',
                    help='draw the per-column entropy envelope behind each '
                         'panel of the grid figure; off by default because at '
                         'panel size it aliases into a grey haze')
    ap.add_argument('--clip-percentile', type=float, default=2.0,
                    help='floor the hairpin and homo-dimer axes at this '
                         'percentile of their pooled values; points below are '
                         'drawn on the floor as hollow triangles and counted. '
                         '0 disables [2]')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--no-separate-panels', action='store_true',
                    help='write only the combined figure')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    design_gc = None
    if args.design_gc.lower() != 'none':
        design_gc = tuple(float(v) for v in args.design_gc.split(','))

    tables, order = {}, []
    for spec in args.primers:
        label, _, path = spec.partition('=')
        if not path:
            label, path = '', label
        for lbl, rows in load_primers(label, path).items():
            lbl = lbl or os.path.basename(os.path.dirname(os.path.dirname(path)))
            tables[lbl] = rows
            if lbl not in order:
                order.append(lbl)

    any_degenerate = any(any(b not in 'ACGTU' for b in s.upper())
                         for rows in tables.values() for _, _, s in rows)
    expand = {'auto': any_degenerate, 'true': True, 'false': False}[args.deg]
    print('degenerate expansion: %s (--deg=%s, IUPAC codes present: %s)'
          % ('on' if expand else 'off', args.deg, any_degenerate))

    # Placement is only needed to recover the opt sequence of a degenerate
    # primer, and loading the MSA class is the expensive half of this script.
    need_consensus = expand and any_degenerate
    print('\nloading %s' % args.msa)
    pct, ent, smooth, msa_gc, shape = entropy_profile(
        args.msa, ignore_ambiguous=args.entropy_ignore_ambiguous)
    print('  %d sequences, %d columns, GC %.1f%%' % (shape[0], shape[1], msa_gc))
    profiles = [(args.msa_label or _pretty(args.msa), pct, ent, smooth, smooth)]
    for spec in args.entropy_msa:
        label, _, path = spec.partition('=')
        if not path:
            label, path = _pretty(label), label
        p, e, s, g, sh = entropy_profile(
            path, ignore_ambiguous=args.entropy_ignore_ambiguous)
        print('  + %s: %d sequences, %d columns, GC %.1f%%' % (label, sh[0], sh[1], g))
        profiles.append((label, p, e, s, s))

    msa = None
    if need_consensus:
        msa = MSA(args.msa, args.threads)
        print('  majority consensus %d nt' % len(msa.consensus))

    jobs = []
    for label in order:
        rows = tables[label]
        arm_jobs, skipped = build_jobs(label, rows, msa, args.max_permutations,
                                       expand, args.min_score_frac)
        print('\n%s: %d primers' % (label, len(rows)))
        for name, why in skipped:
            print('  %s -- %s' % (name, why))
        jobs += arm_jobs

    if not jobs:
        raise SystemExit('no readable primers')

    print('\nscoring %d primers with primer3 (threads=%d)' % (len(jobs), args.threads))
    if args.threads > 1:
        with multiprocessing.Pool(args.threads) as pool:
            rows = pool.map(_worker, jobs, chunksize=8)
    else:
        rows = [_worker(j) for j in jobs]

    df = pd.DataFrame(rows)
    arms = [a for a in order if a in set(df['arm'])]
    df.to_csv(os.path.join(args.out, 'primer_parameters.csv'), index=False)

    capped = int((df['n_permutations'] > df['n_permutations_scored']).sum())
    if capped:
        print('  %d primer(s) had more than %d permutations and were averaged '
              'over the first %d' % (capped, args.max_permutations,
                                     args.max_permutations))

    summary = []
    for arm in arms:
        s = df[df['arm'] == arm]
        row = {'arm': arm, 'n_primers': len(s),
               'mean_len': s['primer_len'].mean(),
               'pct_degenerate': 100.0 * (s['n_permutations'] > 1).mean(),
               'mean_permutations': s['n_permutations'].mean(),
               'max_permutations': int(s['n_permutations'].max()),
               'max_ambiguous_nt': int(s['n_ambiguous'].max()),
               'pct_opt_unplaced': 100.0 * (s['opt_source'] == 'unplaced').mean()}
        for key, lo, hi in (('tm',) + VARVAMP_GATE['tm'],
                            ('gc',) + VARVAMP_GATE['gc']):
            for suffix in ('opt', 'mean'):
                v = s['%s_%s' % (key, suffix)].dropna()
                row['%s_%s' % (key, suffix)] = v.mean()
                row['%s_%s_pct_outside' % (key, suffix)] = (
                    100.0 * (~v.between(lo, hi)).mean() if len(v) else np.nan)
        for key, cap_ in (('hairpin', VARVAMP_GATE['hairpin_tm_max']),
                          ('homodimer', VARVAMP_GATE['homodimer_tm_max'])):
            for suffix in ('opt', 'mean'):
                v = s['%s_%s' % (key, suffix)].dropna()
                row['%s_%s' % (key, suffix)] = v.mean()
                row['%s_%s_pct_over' % (key, suffix)] = (
                    100.0 * (v > cap_).mean() if len(v) else np.nan)
        summary.append(row)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(args.out, 'summary.csv'), index=False)

    draw(df, arms, profiles, args.out, design_gc, msa_gc, config.SEED,
         separate=not args.no_separate_panels,
         clip_percentile=args.clip_percentile,
         ignore_ambiguous=args.entropy_ignore_ambiguous,
         grid_raw=args.grid_raw)

    print('\n--- summary (pct_outside / pct_over are against varVAMP defaults, '
          'not Olivia settings) ---')
    cols = ['arm', 'n_primers', 'pct_degenerate', 'mean_permutations',
            'max_ambiguous_nt', 'tm_opt', 'tm_mean', 'tm_mean_pct_outside',
            'hairpin_mean', 'hairpin_mean_pct_over', 'homodimer_mean',
            'homodimer_mean_pct_over', 'gc_mean', 'gc_mean_pct_outside']
    with pd.option_context('display.width', 220, 'display.max_columns', 60):
        print(summary_df[cols].round(2).to_string(index=False))
    print('\nwrote %s' % args.out)


if __name__ == '__main__':
    main()
