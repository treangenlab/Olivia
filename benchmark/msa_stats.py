#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Alignment statistics and primer placement on an alignment, for the design tables.'''

import os
import re
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# Alphabet

BASES = 'ACGT'
_BIT = dict((b, 1 << i) for i, b in enumerate(BASES))

IUPAC = {
    'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', 'U': 'T',
    'R': 'AG', 'Y': 'CT', 'S': 'CG', 'W': 'AT', 'K': 'GT', 'M': 'AC',
    'B': 'CGT', 'D': 'AGT', 'H': 'ACT', 'V': 'ACG', 'N': 'ACGT',
}

COMPLEMENT = {
    'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'U': 'A',
    'R': 'Y', 'Y': 'R', 'S': 'S', 'W': 'W', 'K': 'M', 'M': 'K',
    'B': 'V', 'V': 'B', 'D': 'H', 'H': 'D', 'N': 'N', '-': '-',
}

GAP = ord('-')

# ASCII code -> 4-bit mask over ACGT. 0 for anything unrecognised, which is how
# gaps and junk characters are told apart from real bases downstream.
CODE_MASK = np.zeros(256, dtype=np.uint8)
for _c, _exp in IUPAC.items():
    _m = 0
    for _b in _exp:
        _m |= _BIT[_b]
    CODE_MASK[ord(_c)] = _m
    CODE_MASK[ord(_c.lower())] = _m

# ASCII code -> index into BASES, or 255 for "not an unambiguous base"
CODE_IDX = np.full(256, 255, dtype=np.uint8)
for _i, _b in enumerate(BASES):
    CODE_IDX[ord(_b)] = _i
    CODE_IDX[ord(_b.lower())] = _i
CODE_IDX[ord('U')] = CODE_IDX[ord('u')] = BASES.index('T')


# number of bases a 4-bit mask stands for: 1 = concrete, >1 = ambiguous, 0 = gap
POPCOUNT = np.array([bin(i).count('1') for i in range(16)], dtype=np.uint8)

# How an ambiguity code in a TARGET sequence is treated.
AMBIGUOUS_MODES = ('resolve', 'drop', 'match', 'mismatch')


def revcomp(seq):
    return ''.join(COMPLEMENT.get(b, 'N') for b in reversed(seq.upper()))


# Alignment

def read_fasta(path):
    ids, seqs, name, buf = [], [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip('\n\r')
            if line.startswith('>'):
                if name is not None:
                    ids.append(name)
                    seqs.append(''.join(buf))
                name, buf = line[1:].strip(), []
            elif line:
                buf.append(line.strip())
    if name is not None:
        ids.append(name)
        seqs.append(''.join(buf))
    return ids, seqs


class Msa(object):
    '''An input alignment, plus the majority consensus every primer is placed in.'''

    def __init__(self, path, gap_thresh=0.5):
        ids, seqs = read_fasta(path)
        if not seqs:
            raise SystemExit('no sequences in %s' % path)
        width = set(len(s) for s in seqs)
        if len(width) != 1:
            raise SystemExit('%s is not an alignment: %d distinct sequence '
                             'lengths (%d..%d)'
                             % (path, len(width), min(width), max(width)))
        self.path = path
        self.ids = ids
        self.matrix = np.frombuffer(''.join(seqs).encode('ascii'),
                                    dtype=np.uint8).reshape(len(seqs), len(seqs[0]))
        self.n_seq, self.n_col = self.matrix.shape

        idx = CODE_IDX[self.matrix]                      # (n, L), 255 = not ACGT
        self.counts = np.zeros((4, self.n_col), dtype=np.int64)
        for b in range(4):
            self.counts[b] = (idx == b).sum(axis=0)
        self.n_called = self.counts.sum(axis=0)          # unambiguous bases per column
        self.gap_freq = (self.matrix == GAP).sum(axis=0) / float(self.n_seq)
        self.common_gap = self.gap_freq >= gap_thresh

        # majority consensus over the columns that survive gap masking
        major = np.argmax(self.counts, axis=0)
        keep = (~self.common_gap) & (self.n_called > 0)
        cols = np.flatnonzero(keep)
        self.consensus = ''.join(BASES[i] for i in major[cols])
        self.cons_idx = major[cols].astype(np.uint8)
        self.c2m = np.append(cols, self.n_col).astype(np.int64)
        self.consensus_len = len(self.consensus)

    # -- alignment-level statistics ----------------------------------------

    def pairwise_identity(self, max_pairs=20000, seed=0):
        '''Mean +/- SD percent identity over sampled sequence pairs.'''
        n = self.n_seq
        if n < 2:
            return float('nan'), float('nan'), 0
        idx = CODE_IDX[self.matrix]
        ok = idx < 4
        rng = np.random.default_rng(seed)
        total = n * (n - 1) // 2
        if total <= max_pairs:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:
            seen, pairs = set(), []
            while len(pairs) < max_pairs:
                i, j = rng.integers(0, n, 2)
                if i == j:
                    continue
                key = (min(i, j), max(i, j))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(key)
        vals = []
        for i, j in pairs:
            both = ok[i] & ok[j]
            m = int(both.sum())
            if m:
                vals.append(float((idx[i][both] == idx[j][both]).sum()) / m)
        vals = np.asarray(vals, dtype=float)
        return 100.0 * vals.mean(), 100.0 * vals.std(ddof=1), len(vals)

    def column_stats(self):
        '''Conservation and entropy over the non-common-gap columns.'''
        keep = (~self.common_gap) & (self.n_called > 0)
        c = self.counts[:, keep].astype(float)
        tot = c.sum(axis=0)
        p = c / tot
        major_freq = p.max(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            logp = np.where(p > 0, np.log(p), 0.0)
        ent = -(p * logp).sum(axis=0) / np.log(4.0)      # normalised Shannon
        return {
            'n_cols_scored': int(keep.sum()),
            'pct_common_gap_cols': 100.0 * float(self.common_gap.mean()),
            'pct_conserved_95': 100.0 * float((major_freq >= 0.95).mean()),
            'pct_conserved_99': 100.0 * float((major_freq >= 0.99).mean()),
            'mean_entropy': float(ent.mean()),
            'median_entropy': float(np.median(ent)),
        }

    # -- primer placement ---------------------------------------------------

    def locate(self, primer, window=None):
        """Best position of `primer` in the consensus, searching both strands."""
        best = None
        for strand, probe in (('+', primer.upper()), ('-', revcomp(primer))):
            k = len(probe)
            if k == 0 or k > self.consensus_len:
                continue
            mask = np.zeros((k, 4), dtype=bool)
            for i, b in enumerate(probe):
                m = CODE_MASK[ord(b)]
                for j in range(4):
                    mask[i, j] = bool(m & (1 << j))
            win = sliding_window_view(self.cons_idx, k)   # (n_win, k)
            mm = (k - mask[np.arange(k)[None, :], win].sum(axis=1)).astype(np.int32)
            if window is not None:
                lo = max(0, int(window[0]))
                hi = min(len(mm), int(window[1]) + 1)
                if hi <= lo:
                    continue
                blocked = np.full(len(mm), k + 1, dtype=np.int32)
                blocked[lo:hi] = mm[lo:hi]
                mm = blocked
            pos = int(np.argmin(mm))
            # second best outside the winner's own footprint
            other = mm.copy()
            other[max(0, pos - k):pos + k] = k + 1
            margin = int(other.min()) - int(mm[pos])
            cand = {'mm': int(mm[pos]), 'start': pos, 'stop': pos + k,
                    'strand': strand, 'margin': margin}
            if best is None or cand['mm'] < best['mm']:
                best = cand
        return best

    def rows_under(self, start, stop):
        '''Each sequence's bases opposite consensus span [start, stop).'''
        block = self.matrix[:, self.c2m[start:stop]]
        return [row.tobytes().decode('ascii').upper() for row in block]


# Mismatch profile of one primer against the alignment

def mismatch_profile(msa, primer, start, stop, strand, ambiguous='resolve'):
    '''Mismatches between one primer and every sequence of the alignment.'''
    if ambiguous not in AMBIGUOUS_MODES:
        raise ValueError('ambiguous must be one of %s, got %r'
                         % (', '.join(AMBIGUOUS_MODES), ambiguous))
    k = len(primer)
    probe = primer.upper() if strand == '+' else revcomp(primer)
    pm = np.array([CODE_MASK[ord(b)] for b in probe], dtype=np.uint8)

    per_seq = np.full(msa.n_seq, np.nan)       # NaN where the sequence is absent
    per_pos = np.zeros(k, dtype=np.int64)      # in probe orientation
    n_skipped = n_gapped = n_dropped = 0
    for i_row, row in enumerate(msa.rows_under(start, stop)):
        m = len(row)
        raw = np.frombuffer(row.encode('ascii'), dtype=np.uint8)
        n_gap = int((raw == GAP).sum())
        if n_gap == m:
            # Sequence absent opposite the primer (partial genome): skip it.
            n_skipped += 1
            continue
        if n_gap:
            n_gapped += 1
        tm = CODE_MASK[raw]
        amb = POPCOUNT[tm] > 1
        disjoint = (tm & pm[:m]) == 0
        if ambiguous == 'drop':
            if amb.any():
                n_dropped += 1
                continue
            bad = disjoint
        elif ambiguous == 'match':
            bad = disjoint & ~amb
        elif ambiguous == 'mismatch':
            bad = disjoint | amb
        else:
            bad = disjoint
        per_seq[i_row] = int(bad.sum())
        per_pos[:m] += bad

    if strand == '-':
        per_pos = per_pos[::-1]                # back to primer orientation
    per_pos = per_pos[::-1]                    # index 0 = 3'-most base

    scored = per_seq[~np.isnan(per_seq)]
    n = len(scored)
    return {
        'n_seq_scored': n,
        'n_seq_absent': n_skipped,
        'n_seq_gapped': n_gapped,
        'n_seq_dropped': n_dropped,
        'mean_mm': float(scored.mean()) if n else float('nan'),
        'pct_seq_with_mm': 100.0 * float((scored > 0).mean()) if n else float('nan'),
        'mm_last5': float(per_pos[:5].sum()) / n if n else float('nan'),
        'mm_pos0': float(per_pos[0]) / n if n else float('nan'),
        'profile': per_pos,
        'per_seq': per_seq,
    }


# Amplicons

NAME_RE = re.compile(
    r'^(?P<prefix>.*?)_(?P<idx>\d+)_(?P<side>LEFT|RIGHT|FW|RV|FORWARD|REVERSE)'
    r'(?:_(?P<alt>\d+))?$', re.IGNORECASE)

LEFT_SIDES = ('LEFT', 'FW', 'FORWARD')


def parse_name(name):
    '''(amplicon_id, is_left) from a primer name, or None if it does not parse.'''
    m = NAME_RE.match(name)
    if not m:
        return None
    return int(m.group('idx')), m.group('side').upper() in LEFT_SIDES


def union_length(spans):
    '''Total length covered by a set of possibly overlapping [start, stop) spans.'''
    if not spans:
        return 0
    spans = sorted(spans)
    total, cur_s, cur_e = 0, spans[0][0], spans[0][1]
    for s, e in spans[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + cur_e - cur_s


def amplicons_from(placed):
    '''Group located primers into amplicons.'''
    groups = {}
    unparsed = 0
    for p in placed:
        parsed = parse_name(p['name'])
        if parsed is None:
            unparsed += 1
            continue
        amp_id, is_left = parsed
        g = groups.setdefault(amp_id, {'L': [], 'R': []})
        g['L' if is_left else 'R'].append(p)

    amp_spans, ins_spans, unpaired = [], [], 0
    for amp_id, g in sorted(groups.items()):
        if not g['L'] or not g['R']:
            unpaired += 1
            continue
        a_start = min(p['start'] for p in g['L'])
        a_stop = max(p['stop'] for p in g['R'])
        i_start = max(p['stop'] for p in g['L'])
        i_stop = min(p['start'] for p in g['R'])
        if a_stop <= a_start:
            unpaired += 1
            continue
        amp_spans.append((a_start, a_stop))
        if i_stop > i_start:
            ins_spans.append((i_start, i_stop))
    return amp_spans, ins_spans, unpaired + unparsed


# Driver: place every primer of a target and cache the result

GEOM_COLS = ['tool', 'amplicon_size', 'pool', 'name', 'direction', 'primer',
             'strand', 'strand_agrees', 'start', 'stop', 'cons_mm',
             'n_seq_scored', 'n_seq_absent', 'n_seq_gapped', 'n_seq_dropped',
             'place_margin', 'repositioned',
             'mean_mm', 'mean_mm_site', 'site', 'pct_seq_with_mm',
             'mm_last5', 'mm_pos0', 'profile']


def _add_site_mismatches(rows, per_seq):
    '''Add the best-of-alternates mismatch count for each primer site.'''
    groups = {}
    for i, r in enumerate(rows):
        parsed = parse_name(r['name'])
        side = 'L' if (parsed and parsed[1]) else ('R' if parsed else '?')
        amp = parsed[0] if parsed else r['name']
        key = (r['tool'], r['amplicon_size'], amp, side)
        groups.setdefault(key, []).append(i)
    for key, idx in groups.items():
        stack = np.vstack([per_seq[i] for i in idx])
        if len(idx) > 1:
            # a sequence absent under every alternate stays absent; nanmin would
            # warn on that all-NaN column, so those columns are dropped first
            usable = ~np.isnan(stack).all(axis=0)
            best = np.full(stack.shape[1], np.nan)
            best[usable] = np.nanmin(stack[:, usable], axis=0)
        else:
            best = stack[0]
        scored = best[~np.isnan(best)]
        val = float(scored.mean()) if len(scored) else float('nan')
        label = '%s|%s|%s|%s' % key
        for i in idx:
            rows[i]['mean_mm_site'] = val
            rows[i]['site'] = label


def _reposition(msa, placed, slack=2.0, verbose=True):
    """Pull primers that landed outside their own amplicon back into it."""
    groups = {}
    for p in placed:
        parsed = parse_name(p['row']['name'])
        amp = parsed[0] if parsed else p['row']['name']
        groups.setdefault((p['row']['tool'], p['row'].get('amplicon_size', ''), amp),
                          []).append(p)

    n_moved = 0
    for key, members in groups.items():
        if len(members) < 2:
            continue
        try:
            amp_size = float(key[1])
        except (TypeError, ValueError):
            amp_size = 1000.0
        width = max(500.0, slack * amp_size)
        anchor = max(members, key=lambda p: (p['hit']['margin'], -p['hit']['mm']))
        a = anchor['hit']['start']
        for p in members:
            if p is anchor:
                continue
            if abs(p['hit']['start'] - a) <= width:
                continue
            hit = msa.locate(p['primer'], window=(a - width, a + width))
            if hit is None:
                continue
            if verbose:
                print('  %-26s %6d -> %6d (mm %d -> %d)'
                      % (p['row']['name'], p['hit']['start'], hit['start'],
                         p['hit']['mm'], hit['mm']), file=sys.stderr)
            p['hit'] = hit
            p['repositioned'] = True
            n_moved += 1
    return n_moved


def place_primers(target, cache=True, ambiguous='resolve', verbose=True,
                  reposition_slack=2.0):
    '''Locate and score every primer of one target.'''
    import pandas as pd

    # the mode changes what the file means, so it is part of the name rather
    # than something a stale cache can hide
    cache_path = os.path.join(target.out_dir, 'primer_geometry_%s.csv' % ambiguous)
    primer_table = target.primer_table
    if cache and os.path.isfile(cache_path):
        if os.path.getmtime(cache_path) >= os.path.getmtime(primer_table):
            if verbose:
                print('  reusing %s' % cache_path, file=sys.stderr)
            return pd.read_csv(cache_path)

    if not os.path.isfile(primer_table):
        raise SystemExit('no primer table at %s' % primer_table)
    df = pd.read_csv(primer_table)
    if verbose:
        print('  loading %s' % target.msa_path, file=sys.stderr)
    msa = Msa(target.msa_path)
    if verbose:
        print('  %d sequences x %d columns, consensus %d nt; placing %d primers'
              % (msa.n_seq, msa.n_col, msa.consensus_len, len(df)), file=sys.stderr)

    # pass 1: place every primer anywhere in the consensus
    placed = []
    for _, r in df.iterrows():
        primer = str(r['primer']).strip().upper()
        hit = msa.locate(primer)
        if hit is None:
            continue
        placed.append({'row': r, 'primer': primer, 'hit': hit, 'repositioned': False})

    # Re-place ambiguous primers that landed far from their own amplicon.
    n_repositioned = _reposition(msa, placed, slack=reposition_slack,
                                 verbose=verbose)

    rows, per_seq = [], []
    for p in placed:
        r, primer, hit = p['row'], p['primer'], p['hit']
        prof = mismatch_profile(msa, primer, hit['start'], hit['stop'],
                                hit['strand'], ambiguous=ambiguous)
        declared = str(r.get('direction', '')).lower()
        expect = '+' if declared.startswith('f') else ('-' if declared.startswith('r') else '')
        rows.append({
            'tool': r['tool'], 'amplicon_size': r.get('amplicon_size', ''),
            'pool': r.get('pool', ''), 'name': r['name'],
            'direction': declared, 'primer': primer,
            'strand': hit['strand'],
            'strand_agrees': (expect == '' or expect == hit['strand']),
            'start': hit['start'], 'stop': hit['stop'], 'cons_mm': hit['mm'],
            'place_margin': hit['margin'], 'repositioned': p['repositioned'],
            'n_seq_scored': prof['n_seq_scored'],
            'n_seq_absent': prof['n_seq_absent'],
            'n_seq_gapped': prof['n_seq_gapped'],
            'n_seq_dropped': prof['n_seq_dropped'],
            'mean_mm': prof['mean_mm'],
            'pct_seq_with_mm': prof['pct_seq_with_mm'],
            'mm_last5': prof['mm_last5'], 'mm_pos0': prof['mm_pos0'],
            'profile': ';'.join(str(int(x)) for x in prof['profile']),
        })
        per_seq.append(prof['per_seq'])

    _add_site_mismatches(rows, per_seq)
    if verbose and n_repositioned:
        print('  repositioned %d ambiguous placement(s)' % n_repositioned,
              file=sys.stderr)
    out = pd.DataFrame(rows, columns=GEOM_COLS)
    out.attrs['consensus_len'] = msa.consensus_len
    if cache:
        out.to_csv(cache_path, index=False)
        if verbose:
            print('  wrote %s' % cache_path, file=sys.stderr)
    return out


def consensus_len_of(target):
    '''Consensus length alone, without re-placing primers.'''
    return Msa(target.msa_path).consensus_len


def summarise_arm(geom, consensus_len):
    '''One row of table 2: amplicon count, coverage and mismatch load for one arm.'''
    spans, inserts, unpaired = [], [], 0
    for _, sub in geom.groupby('amplicon_size', dropna=False):
        placed = sub[['name', 'start', 'stop']].to_dict('records')
        a, i, u = amplicons_from(placed)
        spans.extend(a)
        inserts.extend(i)
        unpaired += u

    mm = geom['mean_mm'].astype(float)
    pct = 100.0 / consensus_len if consensus_len else float('nan')
    return {
        'n_primers': int(len(geom)),
        'n_amplicons': int(len(spans)),
        'n_unpaired': int(unpaired),
        'pct_coverage': union_length(spans) * pct,
        'pct_insert_coverage': union_length(inserts) * pct,
        'pct_span': ((max(e for _, e in spans) - min(s for s, _ in spans)) * pct
                     if spans else 0.0),
        'mean_mm_per_primer': float(mm.mean()) if len(mm) else float('nan'),
        'sd_mm_per_primer': float(mm.std(ddof=1)) if len(mm) > 1 else float('nan'),
        'mean_mm_per_site': float(
            geom.drop_duplicates('site')['mean_mm_site'].astype(float).mean()),
        'n_sites': int(geom['site'].nunique()),
        'alternates_per_site': round(float(len(geom)) / max(1, geom['site'].nunique()), 2),
        'pct_primers_any_mm': 100.0 * float((geom['pct_seq_with_mm'] > 0).mean()),
        'mean_mm_last5': float(geom['mm_last5'].astype(float).mean()),
        'mean_mm_pos0': float(geom['mm_pos0'].astype(float).mean()),
        'mean_primer_len': float(geom['primer'].str.len().mean()),
        'pct_degenerate': 100.0 * float(
            geom['primer'].str.upper().str.contains('[^ACGT]', regex=True).mean()),
        'max_cons_mm': int(geom['cons_mm'].max()) if len(geom) else 0,
    }


def render(rows, columns, headers=None, floatfmt='%.2f'):
    '''Markdown table from a list of dicts. Printed tables go to stdout; the CSV
    written alongside is the one to read programmatically.'''
    headers = headers or dict()
    head = [headers.get(c, c) for c in columns]
    body = []
    for r in rows:
        line = []
        for c in columns:
            v = r.get(c, '')
            if isinstance(v, float):
                v = 'n/a' if v != v else floatfmt % v
            line.append(str(v))
        body.append(line)
    width = [max(len(head[i]), *(len(b[i]) for b in body)) if body else len(head[i])
             for i in range(len(columns))]
    out = ['| ' + ' | '.join(h.ljust(width[i]) for i, h in enumerate(head)) + ' |',
           '|' + '|'.join('-' * (width[i] + 2) for i in range(len(columns))) + '|']
    for b in body:
        out.append('| ' + ' | '.join(b[i].ljust(width[i]) for i in range(len(columns))) + ' |')
    return '\n'.join(out)
