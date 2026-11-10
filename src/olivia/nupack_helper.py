from nupack import *
import math
import os
import atexit
import multiprocessing
from math import ceil
from collections.abc import Callable, Iterable
from time import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import basic
import json

import logging
logger = logging.getLogger('main')
import pandas as pd

from msa_tools import average_scores, expand_degenerate_sequence
import primer_gate

def compute_deltaG(seq1, id1, type1, seq2, id2, type2, temp=60):
    try:
        my_model = Model(material='dna', celsius=temp)
        a = Strand(seq1.upper(), name=f"{id1}_{type1}")
        b = Strand(seq2.upper(), name=f"{id2}_{type2}")
        complex_ab = Complex([a, b])
        my_set = ComplexSet(
            strands={a: 1e-8, b: 1e-8},
            complexes=SetSpec(max_size=0, include=[complex_ab])
        )
        result = complex_analysis(my_set, compute=['pfunc'], model=my_model)
        dG = result[complex_ab].free_energy

        # Constants for equilibrium constant calculation
        R = 1.9872e-3  # kcal/mol/K
        T = 273.15 + temp  # Kelvin
        Keq = math.exp(-dG / (R * T))
        log10_Keq = math.log10(Keq) if Keq > 0 else None

        return {
            "primer1_id": id1,
            "primer1_type": type1,
            "primer1_seq": seq1,
            "primer2_id": id2,
            "primer2_type": type2,
            "primer2_seq": seq2,
            "deltaG": dG,
            "Keq": Keq,
            "log10_Keq": log10_Keq
        }
    except Exception as e:
        return {
            "primer1_id": id1,
            "primer1_type": type1,
            "primer1_seq": seq1,
            "primer2_id": id2,
            "primer2_type": type2,
            "primer2_seq": seq2,
            "deltaG": None,
            "Keq": None,
            "log10_Keq": None,
            "error": str(e)
        }

def mp_wrapper(
    func: Callable, all_args: Iterable, n_cpu: int=1, text: str | None=None, starmap: bool=False, 
    unpack_output: bool=False, silent: bool=False
) -> list:
    """Wrapper for threading (replaced multiprocessing)."""
    tik = time()
    if text and not silent:
        logger.info(f'{text} (threads={n_cpu})')
    
    if n_cpu == 1:
        if starmap:
            func_out = [func(*args) for args in all_args]
        else:
            func_out = [func(args) for args in all_args]
    elif n_cpu > 1:
        # Use ThreadPoolExecutor instead of multiprocessing.Pool
        with ThreadPoolExecutor(max_workers=n_cpu) as executor:
            if starmap:
                # For starmap functionality with threading, we need to wrap the function
                def star_func(args):
                    return func(*args)
                func_out = list(executor.map(star_func, all_args))
            else:
                func_out = list(executor.map(func, all_args))
    else:
        print('n_cpu should be an integer')

    if unpack_output:
        return list(zip(*func_out))
    else:
        return func_out

def calculate_primer_pair(args):
    """Process a single primer pair and return the results"""
    id1, id2, primer1, primer2, type1, type2, temperature = args

    # Handle degenerate bases - calculate average deltaG across all sequence combinations
    seqs1 = primer1 if isinstance(primer1, list) else [primer1]
    seqs2 = primer2 if isinstance(primer2, list) else [primer2]
    
    # deltaG is symmetric, so only unordered pairs are folded; the average is unchanged.
    unordered = {}
    for seq1 in seqs1:
        for seq2 in seqs2:
            unordered[(seq1, seq2) if seq1 <= seq2 else (seq2, seq1)] = None
    dG_of_pair = _batched_dG(list(unordered), temperature)

    total_deltaG = 0
    count = 0

    for seq1 in seqs1:
        for seq2 in seqs2:
            dG = dG_of_pair.get((seq1, seq2) if seq1 <= seq2 else (seq2, seq1))
            if dG is not None:
                total_deltaG += dG
                count += 1

    if count == 0:
        return None
    
    avg_deltaG = total_deltaG / count
    badness_score = max(0, -avg_deltaG)  # Higher score for more negative deltaG
    
    # Find the indices for the matrix (you'll need to track primer IDs to indices)
    # This part needs to be handled in the calling function
    return (id1, id2, avg_deltaG, badness_score)


def calculate_primer_pairs_batch(all_args, n_cpu=1):
    """Score primer pairs together without changing pairwise results."""
    unique_pairs = {}
    expanded = []
    for args in all_args:
        id1, id2, primer1, primer2, type1, type2, temperature = args
        seqs1 = primer1 if isinstance(primer1, list) else [primer1]
        seqs2 = primer2 if isinstance(primer2, list) else [primer2]
        pairs = []
        for seq1 in seqs1:
            for seq2 in seqs2:
                key = (seq1, seq2) if seq1 <= seq2 else (seq2, seq1)
                unique_pairs[key] = None
                pairs.append(key)
        expanded.append((id1, id2, pairs, temperature))

    results = {}
    for temperature in {item[3] for item in expanded}:
        pairs = list(unique_pairs)
        dG_of_pair = _batched_dG(pairs, temperature, n_cpu=n_cpu)
        for pair in pairs:
            results[(temperature, pair)] = dG_of_pair.get(pair)

    output = []
    for id1, id2, pairs, temperature in expanded:
        values = [results[(temperature, pair)] for pair in pairs]
        values = [value for value in values if value is not None]
        if not values:
            output.append(None)
            continue
        avg_deltaG = sum(values) / len(values)
        output.append((id1, id2, avg_deltaG, max(0, -avg_deltaG)))
    return output


# Batched, cached deltaG scoring for the dimer optimization loop.

_dG_cache = {}
# Retained so existing callers of clear_dG_cache() keep working.
_dG_cache_lock = threading.Lock()
_CACHE_MISS = object()


def clear_dG_cache():
    '''Drop the cross-run deltaG cache (not required between arms/amplicon
    sizes -- deltaG for a given sequence pair is deterministic -- but useful
    if memory needs reclaiming between unrelated designs).'''
    _dG_cache.clear()


# Canonical strand orientation.
_CANONICAL_FOLD = True


def set_canonical_folding(flag):
    '''Turn canonical strand orientation on or off for this process.'''
    global _CANONICAL_FOLD
    _CANONICAL_FOLD = bool(flag)


def canonical_folding():
    return _CANONICAL_FOLD


def case_variant_collisions(seqs):
    '''Sequences that denote the same molecule but are spelled with different case.'''
    by_upper = {}
    for seq in seqs:
        by_upper.setdefault(seq.upper(), set()).add(seq)
    return {up: spellings for up, spellings in by_upper.items()
            if len(spellings) > 1}


def _canon_pair(seq1, seq2):
    '''The orientation a pair is folded in. Must agree with _cache_key.'''
    return (seq1, seq2) if seq1 <= seq2 else (seq2, seq1)


def _cache_key(seq1, seq2, temp):
    a, b = (seq1, seq2) if seq1 <= seq2 else (seq2, seq1)
    return (a, b, temp)


def _nupack_dG_chunk(args):
    '''deltaG for a list of pairs in one complex_analysis call.'''
    pairs, temp = args
    try:
        my_model = Model(material='dna', celsius=temp)
        strands_conc = {}
        complex_of_pair = {}
        for seq1, seq2 in pairs:
            if seq1 == seq2:
                # One strand species for a self-pair, so NUPACK applies the homodimer
                # symmetry correction.
                a = Strand(seq1.upper(), name=f"a{len(complex_of_pair)}")
                strands_conc[a] = 1e-8
                complex_of_pair[(seq1, seq2)] = Complex([a, a])
            else:
                a = Strand(seq1.upper(), name=f"a{len(complex_of_pair)}")
                b = Strand(seq2.upper(), name=f"b{len(complex_of_pair)}")
                strands_conc[a] = 1e-8
                strands_conc[b] = 1e-8
                complex_of_pair[(seq1, seq2)] = Complex([a, b])

        my_set = ComplexSet(
            strands=strands_conc,
            complexes=SetSpec(max_size=0, include=list(complex_of_pair.values()))
        )
        result = complex_analysis(my_set, compute=['pfunc'], model=my_model)
        return {pair: result[c].free_energy for pair, c in complex_of_pair.items()}
    except Exception:
        return {}


# Process pool for NUPACK folding (forkserver: NUPACK does not survive fork).

_proc_pool = None
_proc_pool_workers = 0
_proc_pool_lock = threading.Lock()
# Set by set_n_cpu(); the default keeps any caller that never calls it on the
# old serial path rather than silently spinning up a pool.
_N_CPU = 1


def set_n_cpu(n):
    '''Default worker count for the deltaG path, and the point at which the pool is built.
    '''
    global _N_CPU
    _N_CPU = max(1, int(n))
    if _N_CPU > 1:
        _get_proc_pool(_N_CPU)


def shutdown_proc_pool():
    global _proc_pool, _proc_pool_workers
    with _proc_pool_lock:
        if _proc_pool is not None:
            _proc_pool.shutdown(wait=False)
            _proc_pool = None
            _proc_pool_workers = 0


atexit.register(shutdown_proc_pool)


def _get_proc_pool(n_cpu):
    '''The shared pool, created on first use. None means "run serially".'''
    global _proc_pool, _proc_pool_workers
    if n_cpu <= 1:
        return None
    if multiprocessing.current_process().name != 'MainProcess':
        return None  # never nest a pool inside a pool worker
    with _proc_pool_lock:
        if _proc_pool is None:
            workers = max(1, min(int(n_cpu), os.cpu_count() or 1))
            try:
                # No forkserver preload: importing NUPACK into the forkserver deadlocks
                # its workers.
                _proc_pool = ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=multiprocessing.get_context('forkserver'),
                )
                _proc_pool_workers = workers
                logger.info('NUPACK process pool: %d workers' % workers)
            except Exception as e:
                logger.warning('could not start NUPACK process pool (%s); '
                               'falling back to serial deltaG' % e)
                _proc_pool = None
                _proc_pool_workers = 0
        return _proc_pool


def compute_deltaG_batch(pairs, temp=60):
    '''deltaG for many concrete (seq1, seq2) pairs, through the shared cache: {pair: deltaG
    or None}.
    '''
    out = {}
    to_compute = []
    for seq1, seq2 in pairs:
        cached = _dG_cache.get(_cache_key(seq1, seq2, temp), _CACHE_MISS)
        if cached is not _CACHE_MISS:
            out[(seq1, seq2)] = cached
        else:
            to_compute.append((seq1, seq2))

    if not to_compute:
        return out

    computed = _nupack_dG_chunk((to_compute, temp))
    for pair, dG in computed.items():
        _dG_cache[_cache_key(pair[0], pair[1], temp)] = dG
    # Pairs the chunk failed on are reported as None but NOT cached, so a
    # transient failure does not become permanent.
    for pair in to_compute:
        out[pair] = computed.get(pair)
    return out


# Pairs per complex_analysis call (throughput plateaus around here).
_DG_CHUNK = 1024


def _batched_dG(pairs, temperature, n_cpu=None, cache=True):
    '''deltaG for many pairs, keyed exactly as passed.'''
    if n_cpu is None:
        n_cpu = _N_CPU

    canon = _CANONICAL_FOLD
    out = {}
    misses = []
    missing_keys = set()
    # Under canonicalization the folded pair can differ from the requested one; keep the
    # mapping.
    pending = [] if canon else None
    for pair in pairs:
        if canon:
            key_pair = _canon_pair(pair[0], pair[1])
            # already sorted, so this is _cache_key without re-comparing
            key = (key_pair[0], key_pair[1], temperature)
        else:
            key_pair = pair
            key = _cache_key(pair[0], pair[1], temperature)
        if cache:
            cached = _dG_cache.get(key, _CACHE_MISS)
            if cached is not _CACHE_MISS:
                out[pair] = cached
                continue
        if key not in missing_keys:
            misses.append(key_pair)
            missing_keys.add(key)
        if pending is not None:
            pending.append((pair, key_pair))

    # pending is empty whenever misses is: a pair reaches it only by not being
    # a cache hit, and such a pair either starts a miss or shares one's key.
    if not misses:
        return out

    resolved = {} if canon else None
    chunks = [misses[i:i + _DG_CHUNK] for i in range(0, len(misses), _DG_CHUNK)]
    pool = _get_proc_pool(n_cpu) if len(chunks) > 1 else None

    if pool is not None:
        try:
            results = list(pool.map(_nupack_dG_chunk,
                                    [(c, temperature) for c in chunks]))
        except Exception as e:
            logger.warning('NUPACK process pool failed (%s); running serially' % e)
            shutdown_proc_pool()
            results = [_nupack_dG_chunk((c, temperature)) for c in chunks]
    else:
        results = [_nupack_dG_chunk((c, temperature)) for c in chunks]

    for chunk, computed in zip(chunks, results):
        if not computed and len(chunk) > 1:
            # whole-chunk failure: retry per pair so one bad sequence only
            # costs itself
            computed = {}
            for seq1, seq2 in chunk:
                computed[(seq1, seq2)] = compute_deltaG(
                    seq1, None, None, seq2, None, None, temp=temperature)['deltaG']
        if cache:
            for pair, dG in computed.items():
                _dG_cache[_cache_key(pair[0], pair[1], temperature)] = dG
        if canon:
            resolved.update((pair, computed.get(pair)) for pair in chunk)
        else:
            for pair in chunk:
                out[pair] = computed.get(pair)

    if canon:
        # One folded value answers every spelling that reduces to it.
        for pair, key_pair in pending:
            out[pair] = resolved.get(key_pair)
    return out


def batch_badness_vs_many(id1, seqs1, type1, others, temperature, n_cpu=1):
    '''Score one (possibly degenerate) primer against many other primers.'''
    seqs1 = seqs1 if isinstance(seqs1, list) else [seqs1]

    all_pairs = []
    owner = []
    for id2, seqs2, _type2 in others:
        seqs2 = seqs2 if isinstance(seqs2, list) else [seqs2]
        for s1 in seqs1:
            for s2 in seqs2:
                all_pairs.append((s1, s2))
                owner.append(id2)

    if not all_pairs:
        return {}

    # Through _batched_dG for bounded chunks and per-pair retry.
    dG_of_pair = _batched_dG(all_pairs, temperature, n_cpu=n_cpu)

    sums = {}
    counts = {}
    for (s1, s2), id2 in zip(all_pairs, owner):
        dG = dG_of_pair.get((s1, s2))
        if dG is None:
            continue
        sums[id2] = sums.get(id2, 0.0) + dG
        counts[id2] = counts.get(id2, 0) + 1

    return {id2: (total / counts[id2], max(0.0, -(total / counts[id2])))
            for id2, total in sums.items()}


def batch_badness_for_many(requests, temperature, n_cpu=1):
    """Score multiple mutated primers with one shared NUPACK batch."""
    all_pairs = []
    owners = []
    for request_index, (_id1, seqs1, others) in enumerate(requests):
        seqs1 = seqs1 if isinstance(seqs1, list) else [seqs1]
        for _id2, seqs2, _type2 in others:
            seqs2 = seqs2 if isinstance(seqs2, list) else [seqs2]
            for s1 in seqs1:
                for s2 in seqs2:
                    all_pairs.append((s1, s2))
                    owners.append((request_index, _id2))

    if not all_pairs:
        return [{} for _ in requests]

    dG_of_pair = _batched_dG(all_pairs, temperature, n_cpu=n_cpu)
    sums = [{} for _ in requests]
    counts = [{} for _ in requests]
    for pair, (request_index, id2) in zip(all_pairs, owners):
        dG = dG_of_pair.get(pair)
        if dG is None:
            continue
        sums[request_index][id2] = sums[request_index].get(id2, 0.0) + dG
        counts[request_index][id2] = counts[request_index].get(id2, 0) + 1

    return [
        {id2: (total / counts[i][id2], max(0.0, -(total / counts[i][id2])))
         for id2, total in request_sums.items()}
        for i, request_sums in enumerate(sums)
    ]


class primer_generator(object):
    '''Primer candidate generator: g = primer_generator(temperature, salinity); g.get(seq,
    ...).
    '''
    def __init__(self, temperature=60, salinity=0.18, end_pool_path=None):
        paraH = np.array([[-7.6, -7.2, -8.4, -7.8],
                          [-7.2, -7.6, -8.2, -8.5],
                          [-8.5, -7.8, -8.0, -10.6],
                          [-8.2, -8.4, -9.8, -8.0]])
        paraS = np.array([[-21.3, -20.4, -22.4, -21.0],
                          [-21.3, -21.3, -22.2, -22.7],
                          [-22.7, -21.0, -19.9, -27.2],
                          [-22.2, -22.4, -24.4, -19.9]])
        self.paraS = paraS + 0.368 * np.log(salinity)
        self.paraG = paraH - (temperature + 273.15) * self.paraS / 1000
        self.dG_init = 0.2 - (temperature + 273.15) * (-5.7)/1000

        # load pre-generated pool of primer end sequences
        if not end_pool_path is None:
            with open(end_pool_path) as f:
                end_pool = json.load(f)
            self.check_end = True
            self.l_end = len(end_pool[0])
            self.end_pool = [s.lower() for s in end_pool]
        else:
            self.check_end = False
            self.l_end = None
            self.end_pool = None
        
    def StacksDG(self, seq):
        rep = basic.seq2arr(seq)
        dG_stacks = 0
        for i, _ in enumerate(rep[:-1]):
            dG_stacks += self.paraG[rep[i]][rep[i+1]]
        return dG_stacks
    
    """def get(self, seq, prefix='', dG_max=-11.8, min_GC=0.2, max_GC=0.75, temperature=60,
            min_complexity=0.4, min_len=15, max_len=36, check_SNP=True, check_BLAST=False):
        '''
        Input:
            Sequence to generate primers from; restrictions (dG, GC, etc.)
        Output:
            DataFrame of primers; dictionary of failure cases
        '''
        INTR_SCORE = 2

        # input should be lowercase
        if seq == seq.upper(): # all uppercase
            # logger.info('input sequence should be in lowercase, with SNPs in uppercase. ')
            seq = seq.lower()
            if check_SNP:
                # logger.info('ignore check_SNP')
                check_SNP = False

        primer_seq = [] # primer sequence
        primer_arr = [] # array format
        primer_len = [] # primer length without prefix
        primer_dist = [] # distance to mother sequence end
        badness = []
        primer_dG = []
        BLAST_hits = []
        end_fail = 0 # number of primers failed end check
        SNP_fail = 0 # number of primers failed SNP check
        GC_fail = 0 # number of primers failed GC ratio check
        complexity_fail = 0 # number of primers failed complexity check
        for end_pos in range(len(seq)-1, 8, -1): # end_pos is 0-based
            # check end
            if self.check_end:
                if not seq[end_pos-self.l_end+1:end_pos+1] in self.end_pool:
                    end_fail += 1
                    continue
            # check SNP in last 5 bases
            if check_SNP:
                if seq[end_pos-4:end_pos+1] != seq[end_pos-4:end_pos+1].lower():
                    SNP_fail += 1
                    continue
            start_pos = end_pos - 1
            curr_dG = self.dG_init + average_scores(expand_degenerate_sequence(seq[start_pos:end_pos+1]), self.StacksDG)[0] 
            # satisfy minimum length
            while (start_pos > end_pos - min_len + 1) and (start_pos > 0):
                start_pos -= 1
                curr_dG += average_scores(expand_degenerate_sequence(seq[start_pos:(start_pos+2)]), self.StacksDG)[0] 
            # long enough to satisfy energy but not exceeding max_len
            while (curr_dG > dG_max) and (start_pos > end_pos - max_len + 1) and (start_pos > 0):
                start_pos -= 1
                curr_dG += average_scores(expand_degenerate_sequence(seq[start_pos:(start_pos+2)]), self.StacksDG)[0]    
            # check restrictions
            if (curr_dG <= dG_max) or (end_pos - start_pos + 1 == max_len):
                GC_ratio = average_scores([seq[start_pos:end_pos+1]],basic.get_GC)[0]
                if min_GC <= GC_ratio <= max_GC:
                    complexity = average_scores([seq[start_pos:end_pos+1]], basic.get_complexity)[0]
                    if complexity >= min_complexity:
                        s_raw = seq[start_pos:end_pos+1]
                        s = prefix + s_raw

                        s_expand = expand_degenerate_sequence(s)
                        primer_seq.append(s_expand)
                        primer_arr.append([basic.seq2arr(s) for s in s_expand])

                        primer_len.append(end_pos - start_pos + 1)
                        primer_dist.append(len(seq) - end_pos - 1)
                        primer_dG.append(curr_dG)

                        args = (None, None, s_expand, s_expand, None, None, temperature)
                        _, _, _, self_badness = calculate_primer_pair(args)
                        if check_BLAST:
                            #count = BLAST_seq(s_raw, **BLAST_config)
                            if count > 0:
                                #b = INTR_SCORE * WAlignScore(s) + 1000 * np.log10(count)**4
                                b = INTR_SCORE * self_badness + 1000 * np.log10(count)**4
                            else:
                                #b = INTR_SCORE * WAlignScore(s)
                                b = INTR_SCORE * self_badness
                        else:
                            count = -1
                            #b = INTR_SCORE * WAlignScore(s)
                            b = INTR_SCORE * self_badness
                        BLAST_hits.append(count)
                        badness.append(b)
                    else:
                        complexity_fail += 1
                else:
                    GC_fail += 1
        primers = pd.DataFrame({'seq': primer_seq, 
                                'arr': primer_arr, 
                                'primer_len': primer_len, 
                                'dist': primer_dist,    # distance from the primer to the end of seq
                                'badness': badness, 
                                'dG': primer_dG, 
                                'BLAST_hits': BLAST_hits})

        return primers, {'end_fail': end_fail, 'SNP_fail': SNP_fail, 
        'GC_fail': GC_fail, 'complexity_fail': complexity_fail}"""
    
    def get(self, seq, prefix='', dG_max=-11.8, min_GC=0.2, max_GC=0.75, temperature=60,
        min_complexity=0.4, min_len=15, max_len=36, check_SNP=True, check_BLAST=False,
        threads=1, silent=False, gate=None, max_deg=None):
        '''Generate primer candidates from a sequence (max_deg caps degenerate expansions;
        gate adds hard limits).
        '''
        INTR_SCORE = 2

        # input should be lowercase
        if seq == seq.upper(): # all uppercase
            seq = seq.lower()
            if check_SNP:
                check_SNP = False

        # Initialize counters
        end_fail = 0
        SNP_fail = 0
        GC_fail = 0
        complexity_fail = 0
        deg_fail = 0
        # Always present, so the caller's max(fail, key=fail.get) sees the same
        # keys whether or not a gate is active.
        gate_fail = {key: 0 for key in primer_gate.GATE_FAIL_KEYS}

        # Prepare all candidate primers for parallel processing
        all_candidates = []
        
        for end_pos in range(len(seq)-1, 8, -1):
            # check end
            if self.check_end:
                if not seq[end_pos-self.l_end+1:end_pos+1] in self.end_pool:
                    end_fail += 1
                    continue
            
            # check SNP in last 5 bases
            if check_SNP:
                if seq[end_pos-4:end_pos+1] != seq[end_pos-4:end_pos+1].lower():
                    SNP_fail += 1
                    continue
            
            start_pos = end_pos - 1
            curr_dG = self.dG_init + average_scores(expand_degenerate_sequence(seq[start_pos:end_pos+1]), self.StacksDG)[0] 
            
            # satisfy minimum length
            while (start_pos > end_pos - min_len + 1) and (start_pos > 0):
                start_pos -= 1
                curr_dG += average_scores(expand_degenerate_sequence(seq[start_pos:(start_pos+2)]), self.StacksDG)[0] 
            
            # long enough to satisfy energy but not exceeding max_len
            while (curr_dG > dG_max) and (start_pos > end_pos - max_len + 1) and (start_pos > 0):
                start_pos -= 1
                curr_dG += average_scores(expand_degenerate_sequence(seq[start_pos:(start_pos+2)]), self.StacksDG)[0]    
            
            # check restrictions
            if (curr_dG <= dG_max) or (end_pos - start_pos + 1 == max_len):
                GC_ratio = average_scores([seq[start_pos:end_pos+1]], basic.get_GC)[0]
                if min_GC <= GC_ratio <= max_GC:
                    complexity = average_scores([seq[start_pos:end_pos+1]], basic.get_complexity)[0]
                    if complexity >= min_complexity:
                        s_raw = seq[start_pos:end_pos+1]
                        # Reject over-degenerate candidates before any NUPACK work
                        # (self-badness is n**2 folds).
                        if max_deg is not None and basic.get_combinations(s_raw) > max_deg:
                            deg_fail += 1
                            continue
                        # Gate the target-binding part only, without the prefix.
                        gate_reject = gate.check(s_raw) if gate else None
                        if gate_reject is None:
                            all_candidates.append((
                                start_pos, end_pos, s_raw, curr_dG, prefix, temperature, check_BLAST, INTR_SCORE, len(seq)
                            ))
                        else:
                            gate_fail[gate_reject] += 1
                    else:
                        complexity_fail += 1
                else:
                    GC_fail += 1

        # Process candidates in parallel using mp_wrapper
        if all_candidates:
            # Define the processing function that accepts a single tuple
            def process_candidate(candidate_tuple):
                start_pos, end_pos, s_raw, curr_dG, prefix, temp, blast_check, intr_score, seq_length = candidate_tuple
                
                s = prefix + s_raw
                s_expand = expand_degenerate_sequence(s)
                
                # Calculate primer array representation
                primer_arr = [basic.seq2arr(s) for s in s_expand]
                
                # Calculate self-interaction badness
                args = (None, None, s_expand, s_expand, None, None, temp)
                _, _, _, self_badness = calculate_primer_pair(args)
                
                # Handle BLAST checking if enabled
                if blast_check:
                    # count = BLAST_seq(s_raw, **BLAST_config)
                    if count > 0:
                        badness = intr_score * self_badness + 1000 * np.log10(count)**4
                    else:
                        badness = intr_score * self_badness
                else:
                    count = -1
                    badness = intr_score * self_badness
                
                return {
                    'seq': s_expand,
                    'arr': primer_arr,
                    'primer_len': end_pos - start_pos + 1,
                    'dist': seq_length - end_pos - 1,
                    'badness': badness,
                    'dG': curr_dG,
                    'BLAST_hits': count
                }
            
            results = mp_wrapper(
                func=process_candidate,
                all_args=all_candidates,
                n_cpu=threads,
                text='Processing primers',
                silent=silent
            )
        else:
            results = []

        # Convert results to DataFrame
        primer_data = {
            'seq': [], 'arr': [], 'primer_len': [], 'dist': [], 
            'badness': [], 'dG': [], 'BLAST_hits': []
        }
        
        for result in results:
            primer_data['seq'].append(result['seq'])
            primer_data['arr'].append(result['arr'])
            primer_data['primer_len'].append(result['primer_len'])
            primer_data['dist'].append(result['dist'])
            primer_data['badness'].append(result['badness'])
            primer_data['dG'].append(result['dG'])
            primer_data['BLAST_hits'].append(result['BLAST_hits'])

        primers = pd.DataFrame(primer_data)

        return primers, {
            'end_fail': end_fail,
            'SNP_fail': SNP_fail,
            'GC_fail': GC_fail,
            'complexity_fail': complexity_fail,
            'deg_fail': deg_fail,
            **gate_fail
        }