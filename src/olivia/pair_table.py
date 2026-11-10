#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Memoized badness between primer candidates for the dimer optimization.

Each candidate pair's badness (mean deltaG over its concrete expansion pairs) is
computed once and looked up on every annealing trial. deg_sample folds a random
subset of expansion pairs instead of all of them; precompute fills the table in
large batches before annealing.
'''

import zlib
from time import time

import numpy as np

import nupack_helper


HAVE_UNFILLED = 0
HAVE_VALUE = 1
HAVE_EMPTY = 2


class PairTable(object):
    '''Lazily filled, ordered table of candidate-vs-candidate badness.'''

    __slots__ = ('seqs', 'n', 'temperature', 'avg', 'have',
                 'n_filled', 'n_concrete', 'sample_k', 'n_sampled')

    def __init__(self, seq_lists, temperature, sample_k=0):
        self.seqs = [list(s) for s in seq_lists]
        self.n = len(self.seqs)
        self.temperature = temperature
        self.avg = np.zeros((self.n, self.n), dtype=np.float64)
        self.have = np.zeros((self.n, self.n), dtype=np.uint8)
        self.n_filled = 0       # candidate pairs resolved
        self.n_concrete = 0     # concrete sequence pairs folded
        # 0 disables sampling and every expansion pair is folded, which is the exact
        # path.
        self.sample_k = int(sample_k or 0)
        self.n_sampled = 0      # candidate pairs estimated rather than exact

    # -- filling ----------------------------------------------------------

    def _cell_pairs(self, i, j):
        '''The concrete (s1, s2) pairs backing cell (i, j), in accumulation order.'''
        seqs1, seqs2 = self.seqs[i], self.seqs[j]
        m = len(seqs1) * len(seqs2)
        k = self.sample_k
        if not k or m <= k:
            return [(s1, s2) for s1 in seqs1 for s2 in seqs2], False
        # Sample in a canonical frame so cells (i, j) and (j, i) use the same concrete
        # pairs.
        key1, key2 = '\n'.join(seqs1), '\n'.join(seqs2)
        flip = key1 > key2
        if flip:
            key1, key2 = key2, key1
            nb = len(seqs1)
        else:
            nb = len(seqs2)
        rng = np.random.default_rng(zlib.crc32((key1 + '\x00' + key2).encode()))
        # sorted, so accumulation runs in one direction and the k >= m case
        # degrades into the exact path continuously
        idx = np.sort(rng.choice(m, k, replace=False))
        if flip:
            return [(seqs1[f % nb], seqs2[f // nb]) for f in idx], True
        return [(seqs1[f // nb], seqs2[f % nb]) for f in idx], True

    def _fill(self, cells, n_cpu):
        '''Resolve every (i, j) in cells, which must all be unfilled.'''
        if not cells:
            return {}
        all_pairs = []
        cell_pairs = []
        for i, j in cells:
            pairs, sampled = self._cell_pairs(i, j)
            cell_pairs.append(pairs)
            if sampled:
                self.n_sampled += 1
            all_pairs.extend(pairs)

        # The cache can only be bypassed once folding is canonical.
        dG_of_pair = nupack_helper._batched_dG(
            all_pairs, self.temperature, n_cpu=n_cpu,
            cache=not nupack_helper.canonical_folding())
        self.n_concrete += len(all_pairs)

        cache = nupack_helper._dG_cache
        key = nupack_helper._cache_key
        override = {}
        for (i, j), pairs in zip(cells, cell_pairs):
            # Same summation order as nupack_helper.batch_badness_for_many, so exact
            # mode matches it.
            total = trial_total = 0.0
            count = trial_count = 0
            for pair in pairs:
                dG = dG_of_pair.get(pair)
                if dG is not None:
                    trial_total += dG
                    trial_count += 1
                elif pair not in dG_of_pair:
                    # dropped as the reverse of a pair folded in this batch;
                    # the value is now cached under the shared key
                    dG = cache.get(key(pair[0], pair[1], self.temperature))
                if dG is None:
                    continue
                total += dG
                count += 1
            if count:
                self.avg[i, j] = total / count
                self.have[i, j] = HAVE_VALUE
            else:
                self.have[i, j] = HAVE_EMPTY
            if trial_count != count:
                override[(i, j)] = ((HAVE_VALUE, trial_total / trial_count)
                                    if trial_count else (HAVE_EMPTY, 0.0))
            self.n_filled += 1
        return override

    def _cell_size(self, i, j):
        '''How many concrete pairs cell (i, j) will fold.'''
        m = len(self.seqs[i]) * len(self.seqs[j])
        return m if not self.sample_k else min(m, self.sample_k)

    def precompute(self, cells, n_cpu=1, pair_budget=2000000, log=None):
        '''Fill every cell in cells up front, in batches as large as memory allows.'''
        todo, seen = [], set()
        for cell in cells:
            if cell in seen or self.have[cell[0], cell[1]] != HAVE_UNFILLED:
                continue
            seen.add(cell)
            todo.append(cell)
        if not todo:
            return 0

        total = sum(self._cell_size(i, j) for i, j in todo)
        if log:
            log.info('pair table: precomputing %d cells, %d concrete pairs, '
                     '%d workers' % (len(todo), total, n_cpu))
        t0 = time()
        batch, batch_pairs, done = [], 0, 0
        for cell in todo:
            size = self._cell_size(*cell)
            if batch and batch_pairs + size > pair_budget:
                self._fill(batch, n_cpu)
                done += batch_pairs
                if log:
                    el = time() - t0
                    log.info('  %6.1f%%  %d/%d pairs  %.0f pairs/s'
                             % (100.0*done/total, done, total,
                                done/el if el > 0 else 0))
                batch, batch_pairs = [], 0
            batch.append(cell)
            batch_pairs += size
        if batch:
            self._fill(batch, n_cpu)
            done += batch_pairs
        el = time() - t0
        if log:
            log.info('pair table: filled %d cells, %d pairs in %.1fs '
                     '(%.0f pairs/s)'
                     % (len(todo), done, el, done/el if el > 0 else 0))
        return len(todo)

    def ensure(self, cells, n_cpu=1):
        '''Fill any of cells not already present.'''
        todo = [(i, j) for i, j in cells
                if self.have[i, j] == HAVE_UNFILLED]
        if not todo:
            return {}
        # de-duplicate while keeping order, so one batch never folds the
        # same candidate pair twice
        seen = set()
        unique = []
        for cell in todo:
            if cell not in seen:
                seen.add(cell)
                unique.append(cell)
        return self._fill(unique, n_cpu)

    # -- reading ----------------------------------------------------------

    def badness_rows(self, requests, n_cpu=1):
        '''Score mutated primers against their partners.'''
        cells = []
        for i, partners in requests:
            for j in partners:
                cells.append((i, j))
        override = self.ensure(cells, n_cpu=n_cpu)

        out = []
        for i, partners in requests:
            row = self.avg[i]
            have = self.have[i]
            scored = []
            for pos, j in enumerate(partners):
                if override:
                    state, value = override.get((i, j), (have[j], row[j]))
                else:
                    state, value = have[j], row[j]
                if state == HAVE_VALUE:
                    scored.append((pos, max(0.0, -value)))
            out.append(scored)
        return out


def intern_candidates(seq_lists):
    '''Map candidate sequence lists to integer ids, deduplicating by content.'''
    index = {}
    unique = []
    ids = []
    for seqs in seq_lists:
        key = tuple(seqs)
        got = index.get(key)
        if got is None:
            got = len(unique)
            index[key] = got
            unique.append(list(seqs))
        ids.append(got)
    return ids, unique


def pool_cells(curr_tube, opt_cand):
    '''Every candidate pair the annealing can reach in one pool.'''
    slots = []
    for plex_id in curr_tube:
        pairs = opt_cand[plex_id]
        slots.append(sorted({a for a, _ in pairs}))
        slots.append(sorted({b for _, b in pairs}))
    cells = []
    for a in range(len(slots)):
        for b in range(len(slots)):
            if a == b:
                continue
            for i in slots[a]:
                for j in slots[b]:
                    cells.append((i, j))
    return cells
