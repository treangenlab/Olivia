import logging
logger = logging.getLogger('main')

import numpy as np
import pandas as pd
import pickle
import random
import multiprocessing

from time import time
from numpy.random import rand, default_rng
from copy import deepcopy
from tqdm import tqdm
from math import floor

# imternal modules
import basic
from design import PrimerSetBadnessFast
# from msa_tools import Primer, expand_degenerate_sequence
# from collections import defaultdict

import nupack_helper
import pair_table
import primer_gate
from nupack_helper import calculate_primer_pair, calculate_primer_pairs_batch, mp_wrapper

# algorithmic parameters
PRIMER_DESIGN_LEN = 40 # length of primer design region [40]
UPPER_GC = 0.75 # extreme GC upper bond [0.75]
LOWER_GC = 0.25 # extreme GC lower bond [0.75]
LOW_COMPLEXITY = 0.4 # low complexity lower bond [0.4]
CHOICE_RATE = 0.3 # bottom CHOICE_RATE lowest risk primer design regions are choosen from candidate region
RISK_TH = 0.1 # top RISK_TH highest risk primer design regions are considered as loss
MAX_RELAX_ROUNDS = 20 # relaxation steps get_primer() spends on one plex before dropping the primer gate [20]

N_POOLS = 2 # number of primer pool (not to be changed)

# Starting SA temperature for optimize(), as a fraction of the pool's initial loss.
SA_TEMP_FRAC = 0.01

from collections.abc import Callable, Iterable

import gzip, shutil, logging, datetime, subprocess, multiprocessing
from pathlib import Path
from time import time
from io import StringIO
from typing import Literal
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence, Hashable
from tqdm import tqdm  # Import tqdm

import threading
from concurrent.futures import ThreadPoolExecutor

def _sa_budget(config, pool_size):
    '''Resolve the simulated-annealing budget.'''
    num_steps = config.get('NumSteps')
    if not num_steps:
        num_steps = 10 + int(pool_size/10)
    zero_steps = config.get('ZeroSteps')
    if zero_steps is None:
        zero_steps = num_steps
    time_per_step = config.get('TimePerStep')
    if not time_per_step:
        time_per_step = 1000
    return int(num_steps), int(zero_steps), int(time_per_step)


def find_min_loc(risk_arr, start, stop, rng):
    '''
    find the next primer design region within start and stop.
    '''
    score = []
    loc = np.array(range(start, stop-PRIMER_DESIGN_LEN+2))
    for i in loc:
        score.append(sum(risk_arr[i-1: i-1+PRIMER_DESIGN_LEN]))
    if len(loc) == 1:
        idx_rand = 0
    elif CHOICE_RATE < 1:
        k = int(len(loc)*CHOICE_RATE)
        if k == 0:
            k = 1
        idx_k_lowest = np.argpartition(score, k)[:k]
        idx_rand = rng.choice(idx_k_lowest)
    else:
        idx_rand = rng.choice(range(len(loc)))
    primer_start = loc[idx_rand]
    return primer_start, primer_start+PRIMER_DESIGN_LEN-1, score[idx_rand]


def generate_context(SOI):
    '''
    find a valid set of PDRs that cover the whole sequence of interest (SOI).
    '''
    risk_arr, start, stop, max_amp_len, min_amp_len, seed = SOI
    SOI_len = stop - start + 1
    rng = default_rng(seed)
    
    all_context_seq = [] # coordinates of all context sequences
    all_risk = []

    # generate the first pair of primer design region
    fp_start, fp_stop, fp_risk = find_min_loc(risk_arr, start, start+3*PRIMER_DESIGN_LEN-1, rng)
    rp_stop, rp_start, rp_risk = find_min_loc(risk_arr, fp_start+min_amp_len-PRIMER_DESIGN_LEN, fp_start+max_amp_len-1, rng)
    all_context_seq.append((fp_start, fp_stop, rp_stop, rp_start))
    all_risk.append((fp_risk, rp_risk))

    # generate the second pair of primer design region
    left_lim = max(fp_stop+1, rp_start+2*PRIMER_DESIGN_LEN-max_amp_len+1) # leave space for the next fP design region
    fp_start, fp_stop, fp_risk = find_min_loc(risk_arr, left_lim, rp_stop-1, rng)
    left_lim = max(fp_start+min_amp_len-PRIMER_DESIGN_LEN, rp_start+PRIMER_DESIGN_LEN+1)
    right_lim = fp_start+max_amp_len-1
    if right_lim > stop:
        raise ValueError('sequence provided is too short.')
    rp_stop, rp_start, rp_risk = find_min_loc(risk_arr, left_lim, right_lim, rng)
    all_context_seq.append((fp_start, fp_stop, rp_stop, rp_start))
    all_risk.append((fp_risk, rp_risk))

    # design the rest context sequences
    cs_id = 2 # id of the next context sequence to be designed (0-based)
    while True:
        # fp
        # all_context_seq[cs_id-2][3]: rp_start of the last last context sequence
        left_lim = max(all_context_seq[cs_id-2][3]+1, rp_start+2*PRIMER_DESIGN_LEN-max_amp_len+1)
        fp_start, fp_stop, fp_risk = find_min_loc(risk_arr, left_lim, rp_stop-1, rng)
        # rp
        left_lim = max(fp_start+min_amp_len-PRIMER_DESIGN_LEN, rp_start+PRIMER_DESIGN_LEN+1)
        right_lim = fp_start+max_amp_len-1
        if right_lim > stop:
            break
        rp_stop, rp_start, rp_risk = find_min_loc(risk_arr, left_lim, right_lim, rng)

        all_context_seq.append((fp_start, fp_stop, rp_stop, rp_start))
        all_risk.append((fp_risk, rp_risk))
        cs_id += 1

    all_context_seq = np.array(all_context_seq)
    all_risk = np.array(all_risk)
    all_risk_flatten = all_risk.flatten()
    k = int(len(all_risk_flatten) * (1-RISK_TH))

    cover_start = all_context_seq[0][1]+1
    cover_stop = all_context_seq[-1][2]-1
    coverage = (cover_stop-cover_start+1)/SOI_len

    loss = sum(np.partition(all_risk_flatten, k)[k:]**2)/(coverage**2)
    #loss = sum(all_risk_flatten[all_risk_flatten>=RISK_TH])
    return all_context_seq, all_risk, loss


def design_context_seq(config, deg: bool):
    '''PDR optimization.'''
    ref_path = config['ref_path']
    max_amp_len = config['max_amp_len']
    w_gc = config['w_egc']
    w_complexity = config['w_lc']
    w_hits = config['w_ns']
    w_var = config['w_var']

    w_sensi = config['w_sensi']
    w_combi = config['w_combi']

    seed = config['seed']
    n_cpu = config['threads']
    iterMul = config['iterMul']

    if not deg:
        w_sensi = 0
        w_combi = 0
    
    # set random number generator
    rng_parent = default_rng(seed)
    
    with open(ref_path,  'rb') as f:
        olv_ref = pickle.load(f)

    seq_raw = olv_ref['seq'] # SNPs are capitalized in build()
    seq_id = olv_ref['seq_id'] # ID of the fasta record (everything before the space character in the FASTA header)
    start = olv_ref['start'] # start of design region, 1-based, closed
    stop = olv_ref['stop'] # stop of design region, 1-based, closed
    gc_arr = olv_ref['gc_arr']
    comp_arr = olv_ref['comp_arr']
    var_arr = olv_ref['var_arr']
    hits_arr = olv_ref['hits_arr']

    sensi_arr = olv_ref['sensi_arr']
    combi_arr = olv_ref['combi_arr']

    logger.info('Successfully loaded reference file %s' % ref_path)
    
    if not config['min_amp_len']:
        min_amp_len = int(max_amp_len*0.9)
    else:
        min_amp_len = config['min_amp_len']
    if min_amp_len < 3*PRIMER_DESIGN_LEN:
        raise ValueError('min_amp_len is too small')

    gc_arr = np.logical_or(gc_arr<LOWER_GC, gc_arr>UPPER_GC).astype(int)
    comp_arr = (comp_arr < LOW_COMPLEXITY).astype(int)
    
    if max(hits_arr) != 0:
        hits_arr = hits_arr/max(hits_arr) # normalize

    # make arrays the same length as seq_raw
    gc_arr = w_gc*0.5*np.concatenate((np.zeros(start-1), gc_arr, np.zeros(len(seq_raw)-stop)))
    comp_arr = w_complexity*0.5*np.concatenate((np.zeros(start-1), comp_arr, np.zeros(len(seq_raw)-stop)))
    hits_arr = w_hits*np.concatenate((np.zeros(start-1), hits_arr, np.zeros(len(seq_raw)-stop)))
    var_arr = w_var*10*np.concatenate((np.zeros(start-1), var_arr, np.zeros(len(seq_raw)-stop)))

    sensi_arr = w_sensi*0.1*np.concatenate((np.zeros(start-1), sensi_arr, np.zeros(len(seq_raw)-stop)))
    combi_arr = w_combi*np.concatenate((np.zeros(start-1), combi_arr, np.zeros(len(seq_raw)-stop)))

    # construct risk array (first row is risk, second row is coordinate on seq_rawy)
    risk_arr = gc_arr + comp_arr + hits_arr + var_arr + sensi_arr + combi_arr

    N = iterMul * 500*len(risk_arr)//max_amp_len # number of primer sets to generate
    #N = 100
    rand_int = rng_parent.integers(2**32, size=N) # random seeds for each iteration

    # single thread
    # design = [generate_context((risk_arr, start, stop, max_amp_len, min_amp_len, rand_int[i])) for i in tqdm(range(N))]

    # multi threads
    logger.info('reference sequence length: %d' % len(seq_raw))
    logger.info('design region: %d:%d' % (start, stop))
    logger.info('region length: %d' % (stop-start+1))
    logger.info('number of PDR sets to be tested: %d' % N)
    logger.info('Designing PDRs...')
    tik = time()
    start_method = 'forkserver' if 'forkserver' in multiprocessing.get_all_start_methods() else 'spawn'
    with multiprocessing.get_context(start_method).Pool(processes=n_cpu) as pool:
        batch = [(risk_arr, start, stop, max_amp_len, min_amp_len, rand_int[i]) for i in range(N)]
        design = list(
            tqdm(
                pool.imap(generate_context, batch, chunksize=max(N//n_cpu//10, 1)), # chunksize=1 is slow
                total=N, 
                ascii=' >'
            )
        )
    logger.info('Finished in %.3fs' % (time()-tik))

    # Loss of all iterations
    all_loss = [d[2] for d in design]
    all_loss.sort(reverse=True)

    # find the best arangement of primer design regions
    best_design = sorted(design, key=lambda x:x[2])
    all_context_seq, all_risk, loss = best_design[0]
    logger.info('Loss of the best design: %.3f' % loss)
    
    # prepare output
    all_plex_info = {}
    for i, (context_seq, risk) in enumerate(zip(all_context_seq, all_risk)):
        cs = seq_raw[context_seq[0]-1:context_seq[3]] # actual context sequence
        plex_info = {
            'reference': seq_id, 
            'tube': i%2 + 1, 
            'context_seq_coords': (context_seq[0], context_seq[3]), 
            'primers_coords': tuple(context_seq), 
            'insert_coords': (context_seq[1]+1, context_seq[2]-1), 
            'context_seq': cs, 
            'context_seq_revcomp': basic.revcomp(cs), 
            'fP_design': seq_raw[context_seq[0]-1:context_seq[1]], 
            'rP_design': basic.revcomp(seq_raw[context_seq[2]-1:context_seq[3]]), 
            'risk': tuple(risk)
        }
        all_plex_info['%s_%d' % (seq_id, i+1)] = plex_info
    logger.info('total amplicons: %d' % (i+1))
    cover_start = all_context_seq[0][1]+1
    cover_stop = all_context_seq[-1][2]-1
    logger.info('covered region: %d:%d' % (cover_start, cover_stop))
    logger.info('coverage of reference sequence: %.3f%%' % (100*(cover_stop-cover_start+1)/len(seq_raw)))
    return all_plex_info, risk_arr, gc_arr, comp_arr, hits_arr, var_arr, sensi_arr, combi_arr, all_loss, olv_ref['seq_record']

def get_primer(all_plex_info, config):
    '''
    Generate candidates of fP and rP for each plex
    Input:
        all_plex_info: output of design_context_seq()
        config: design configurations
    Output:
        all_plex_info: add primer candidates to input all_plex_info
    '''
    logger.info('Generating primer candidates...')
    tik = time()

    temperature = config["temperature"]

    temperature_fP = config['temperature']
    temperature_rP = config['temperature']
    salinity = config['salinity']
    fP_adapter = config['fP_prefix']
    rP_adapter = config['rP_prefix']
    default_fP_setting = {
        'dG_max': config['dG_max'],
        'min_GC': config['min_GC'],
        'max_GC': config['max_GC'],
        'min_complexity': config['min_complexity'],
        'max_len': config['max_len'],
        'check_SNP': config['check_SNP'],
        'max_deg': config['max_deg']
    }

    # Optional varVAMP-shaped hard gate (primer_gate.py).
    gate = primer_gate.PrimerGate.from_spec(config.get('primer_gate'))
    if gate is not None:
        logger.info('Primer gate active: %s' % gate.describe())
        default_fP_setting.update(gate.olivar_overrides())

    default_rP_setting = deepcopy(default_fP_setting)
    check_BLAST = False

    # generate primer candidates
    fP_generator = nupack_helper.primer_generator(temperature_fP, salinity)
    rP_generator = nupack_helper.primer_generator(temperature_rP, salinity)
    n = 0

    def generate(generator, design_seq, prefix, setting, label, plex_id):
        '''Candidates for one direction of one plex, relaxing until there is at least one.
        '''
        local_gate = gate
        result, fail = generator.get(
            design_seq, prefix, temperature=temperature, check_BLAST=check_BLAST,
            **setting, threads=config['threads'], silent=True, gate=local_gate)

        rounds = 0
        while len(result) == 0:
            if rounds >= MAX_RELAX_ROUNDS:
                if local_gate is None:
                    logger.warning(
                        'No %s candidate for %s after %d relaxation rounds; '
                        'leaving this plex without candidates'
                        % (label, plex_id, rounds))
                    break
                logger.warning(
                    'No %s candidate for %s after %d relaxation rounds; '
                    'dropping the primer gate for this plex' % (label, plex_id, rounds))
                local_gate = None
                # GC is the only argument the gate overrode, so it is the only
                # one to restore -- min_len/max_len were never touched.
                setting.update({'min_GC': config['min_GC'], 'max_GC': config['max_GC']})
                rounds = 0
            else:
                # detect failure mode with the most cases
                fail_mode = max(fail, key=fail.get)
                # Log which axis was relaxed.
                logger.info('Fail to generate %s, update setting (%s): %s'
                            % (label, fail_mode, plex_id))
                # update setting
                if fail_mode == 'SNP_fail' and setting['check_SNP'] == True:
                    setting['check_SNP'] = False
                elif fail_mode == 'GC_fail':
                    setting['min_GC'] -= 0.05
                    setting['max_GC'] += 0.05
                elif fail_mode == 'complexity_fail':
                    setting['min_complexity'] -= 0.05
                elif fail_mode == 'deg_fail':
                    # The degeneracy cap is a cost guard: relax it rather than leave a
                    # region undesignable.
                    setting['max_deg'] *= 4
                elif local_gate is not None and fail_mode in primer_gate.GATE_FAIL_KEYS:
                    relaxed = local_gate.relaxed(fail_mode)
                    if relaxed is None:
                        # this axis is already vacuous -- only the round bound
                        # can resolve it, so spend rounds rather than spin
                        rounds = MAX_RELAX_ROUNDS
                        continue
                    local_gate = relaxed
                rounds += 1
            # generate primers again
            result, fail = generator.get(
                design_seq, prefix, temperature=temperature, check_BLAST=check_BLAST,
                **setting, threads=config['threads'], silent=True, gate=local_gate)

        return result, setting

    undesignable = []

    for plex_id, plex_info in tqdm(all_plex_info.items(), ascii=' >'):
        # set primer prefix
        fP_prefix = fP_adapter
        rP_prefix = rP_adapter
        n += 1

        # generate fP
        fP_design = plex_info['fP_design'] # sequence to generate fP candidates
        fP, fP_setting = generate(fP_generator, fP_design, fP_prefix,
                                  deepcopy(default_fP_setting), 'fP', plex_id)
        plex_info['fP_candidate'] = fP      # each fp is a primer list
        plex_info['fP_setting'] = fP_setting

        # generate rP
        rP_design = plex_info['rP_design'] # sequence to generate rP candidates
        rP, rP_setting = generate(rP_generator, rP_design, rP_prefix,
                                  deepcopy(default_rP_setting), 'rP', plex_id)
        plex_info['rP_candidate'] = rP      # each rp is a primer list
        plex_info['rP_setting'] = rP_setting

        # No candidates even after relaxation: drop the plex (an amplicon gap).
        if len(fP) == 0 or len(rP) == 0:
            undesignable.append(plex_id)

    for plex_id in undesignable:
        del all_plex_info[plex_id]
    if undesignable:
        logger.warning('Dropped %d plex(es) with no primer candidates: %s'
                       % (len(undesignable), ', '.join(str(p) for p in undesignable)))

    logger.info('Finished in %.3fs' % (time()-tik))
    return all_plex_info

def optimize(all_plex_info, config):
    '''Optimize fP and rP with the NUPACK algorithm.'''
    logger.info('Optimizing primer dimers...')
    # Score a plex's own fP-rP dimer for the proposed pair (False: Olivar 1.3.3's legacy
    # scoring).
    fix_intraplex = config.get('fix_intraplex', True)
    if not fix_intraplex:
        logger.warning('intra-plex fP-rP dimer scoring: legacy (known defect)')
    # set random seed
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    nupack_helper.set_n_cpu(config['threads'])

    # automatically determine # SADDLE iterations
    pool_size = max(len(all_plex_info)/N_POOLS - 100, 0)
    # InitSATemp is set per pool from that pool's initial loss, see below.
    NUMSTEPS, ZEROSTEPS, TimePerStep = _sa_budget(config, pool_size)
    temperature = config["temperature"]

    # generate primer pair candidates (fP-rP)
    n_pair = []
    for plex_id, plex_info in all_plex_info.items():
        optimize = []
        plex_amp_len = []
        for i, fp in plex_info['fP_candidate'].iterrows():
            for j, rp in plex_info['rP_candidate'].iterrows():
                insert_len = plex_info['insert_coords'][1] - plex_info['insert_coords'][0] + 1
                # check amplicon length
                amp_len = fp['primer_len'] + fp['dist'] + insert_len + rp['dist'] + rp['primer_len']
                plex_amp_len.append(amp_len)
                optimize.append([fp, rp])
        plex_info['optimize'] = optimize
        n_pair.append(len(optimize))
    logger.info('total primer pairs %d' % sum(n_pair))
    logger.info('average pairs per plex %.2f' % (sum(n_pair)/len(n_pair)))

    # Intern every candidate primer to an integer id, and build the memo table the
    # annealing loop scores out of.
    cand_seq_lists = []
    opt_cand = {}       # plex_id -> [(fP candidate id, rP candidate id), ...]
    for plex_id, plex_info in all_plex_info.items():
        for fp, rp in plex_info['optimize']:
            for row in (fp, rp):
                seq = row['seq']
                cand_seq_lists.append(seq if isinstance(seq, list) else [seq])
        opt_cand[plex_id] = [None] * len(plex_info['optimize'])
    cand_ids, unique_cands = pair_table.intern_candidates(cand_seq_lists)
    k = 0
    for plex_id in all_plex_info:
        rows = opt_cand[plex_id]
        for n in range(len(rows)):
            rows[n] = (cand_ids[k], cand_ids[k+1])
            k += 2
    deg_sample = int(config.get('deg_sample') or 0)
    # Precompute only under canonical folding, where fill order cannot change results.
    precompute_pairs = bool(config.get('precompute_pairs', True)) \
        and bool(config.get('canonical_folding'))
    # Concrete pairs per precompute batch: the knob that bounds peak memory
    # without giving up the batching that makes precompute fast.
    precompute_batch = int(config.get('precompute_batch') or 2000000)
    pair_tbl = pair_table.PairTable(unique_cands, temperature,
                                    sample_k=deg_sample)
    logger.info('candidate primers: %d distinct over %d pair slots'
                % (len(unique_cands), len(cand_seq_lists)//2))
    if deg_sample:
        # Sampled degenerate folding gives an estimated badness; say so in the log.
        logger.info('degenerate sampling: at most %d expansion pairs per '
                    'candidate pair (badness is estimated, not exact)'
                    % deg_sample)
    else:
        logger.info('degenerate sampling: off (all expansion pairs folded)')
    if precompute_pairs:
        logger.info('pair table: precompute on (exact; batches the folds)')
    elif config.get('precompute_pairs', True):
        logger.info('pair table: precompute off (needs canonical folding)')
    else:
        logger.info('pair table: precompute off (lazy per-trial fill)')
    # Raw-string canonicalization assumes one molecule is never spelled with two case
    # patterns.
    collisions = nupack_helper.case_variant_collisions(
        seq for cand in unique_cands for seq in cand)
    if collisions:
        logger.warning('%d sequence(s) spelled with more than one case '
                       'pattern; strand-order canonicalization sorts raw '
                       'strings, so these fold inconsistently' % len(collisions))
    
    # optimize each tube
    all_lc = []
    tik = time()
    for i_tube in range(1, N_POOLS+1):
        logger.info('\npool %d, simulated annealing...' % i_tube)
        # plex_id of current tube
        curr_tube = [plex_id for plex_id, plex_info in all_plex_info.items() if plex_info['tube'] == i_tube]

        # Fill the whole table before annealing, in large batches, rather than a few
        # cells per trial.
        if precompute_pairs:
            pair_tbl.precompute(pair_table.pool_cells(curr_tube, opt_cand),
                                n_cpu=config['threads'],
                                pair_budget=precompute_batch, log=logger)

        # load existing primers
        # existing_primer = config['existing_primer']
        existing_primer = []

        # randomly select one primer set (one pair each plex)
        curr_index = {}
        curr_fp = {}
        curr_rp = {}
        # Candidate id currently selected for each primer slot.
        curr_cand = {}
        curr_badness = 0
        
        # Initialize inter-badness matrix
        all_primers = {}  # Store all primers with their IDs, sequences, and types
        primer_ids = []

        for plex_id in curr_tube:
            # randomly select one primer pair
            i = floor(rand() * len(all_plex_info[plex_id]['optimize']))
            curr_index[plex_id] = i
            fp = all_plex_info[plex_id]['optimize'][i][0]
            rp = all_plex_info[plex_id]['optimize'][i][1]

            # Handle degenerate bases - get all possible sequences
            if isinstance(fp['seq'], list):
                fp_seqs = fp['seq']
            else:
                fp_seqs = [fp['seq']]
            
            if isinstance(rp['seq'], list):
                rp_seqs = rp['seq']
            else:
                rp_seqs = [rp['seq']]

            curr_fp[plex_id] = fp_seqs
            curr_rp[plex_id] = rp_seqs
            curr_cand['fp_%s' % plex_id] = opt_cand[plex_id][i][0]
            curr_cand['rp_%s' % plex_id] = opt_cand[plex_id][i][1]

            curr_badness += fp['badness'] + rp['badness']   # self badness adds-on

            # Add primers to the all_primers dictionary
            fp_id = f'fp_{plex_id}'
            rp_id = f'rp_{plex_id}'
            all_primers[fp_id] = {
                'seq': fp_seqs, 
                'type': 'fP', 
                'id': fp_id,
                'plex_id': plex_id
            }
            all_primers[rp_id] = {
                'seq': rp_seqs, 
                'type': 'rP', 
                'id': rp_id,
                'plex_id': plex_id
            }
            primer_ids.extend([fp_id, rp_id])

        # Create a mapping from primer ID to matrix index
        primer_id_to_index = {primer_id: idx for idx, primer_id in enumerate(primer_ids)}
        n_primers = len(primer_ids)
        # "every primer but this one", in primer_ids order -- the order the
        # per-trial list comprehensions used to rebuild 40,000 times over.
        others_ids = {pid: [q for q in primer_ids if q != pid]
                      for pid in primer_ids}
        others_idx = {pid: [primer_id_to_index[q] for q in others_ids[pid]]
                      for pid in primer_ids}

        # A trial replaces both primers of a plex, so score the sibling as part of the
        # proposed pair.
        sibling = {}
        for plex_id in curr_tube:
            sibling['fp_%s' % plex_id] = 'rp_%s' % plex_id
            sibling['rp_%s' % plex_id] = 'fp_%s' % plex_id
        others_ids_ns = {pid: [q for q in primer_ids
                               if q != pid and q != sibling[pid]]
                         for pid in primer_ids}
        others_idx_ns = {pid: [primer_id_to_index[q] for q in others_ids_ns[pid]]
                         for pid in primer_ids}
        inter_badness_matrix = np.zeros((n_primers, n_primers))
        # deltaG_matrix = np.full((n_primers, n_primers), np.nan)

        # Prepare all the pairs to be processed
        all_pairs = []
        for i, id1 in enumerate(primer_ids):
            primer1 = all_primers[id1]
            for j, id2 in enumerate(primer_ids):
                if i < j:
                    primer2 = all_primers[id2]
                    all_pairs.append((id1, id2, primer1['seq'], primer2['seq'], primer1['type'], primer2['type'], temperature))

        # Use the modified mp_wrapper with progress bar
        logger.info('Calculating interactions (threads=%d)' % config['threads'])
        results = calculate_primer_pairs_batch(all_pairs, n_cpu=config['threads'])
        
        # Process the results and populate your matrices
        for result, _ in zip(results, all_pairs):
            if result is not None:
                id1, id2, _, badness_score = result
                
                # Get matrix indices
                i = primer_id_to_index[id1]
                j = primer_id_to_index[id2]

                inter_badness_matrix[i, j] += badness_score
                inter_badness_matrix[j, i] += badness_score

        # Calculate total inter-badness from the matrix
        inter_badness = np.sum(inter_badness_matrix) / 2
        
        curr_badness += inter_badness
        logger.info('initial loss = %.3f' % curr_badness)

        # Scale the starting temperature to this pool's loss instead of using the stock
        # constant.
        InitSATemp = max(1e-9, SA_TEMP_FRAC * curr_badness)

        # optimization
        learning_curve = []
        SATemp = InitSATemp
        for step in range(NUMSTEPS + ZEROSTEPS):
            # logger.info('SA temperature = %.3f, SADDLE Loss = %.3f' % (SATemp, curr_badness))
            logger.info('SA temperature = %.3f, NUPACK Loss = %.3f' % (SATemp, curr_badness))
            for t in range(TimePerStep):
                # pick a plex to change primer pair
                mutplex = random.choice(curr_tube) # randomly select a plex_id in current tube
                n_pair = len(all_plex_info[mutplex]['optimize'])
                if n_pair == 1:
                    continue
                mutindex = floor(rand() * n_pair) # index of new primer pair

                # copy current design
                new_index = deepcopy(curr_index)
                new_fp = deepcopy(curr_fp)
                new_rp = deepcopy(curr_rp)
                
                # calculate new badness
                new_index[mutplex] = mutindex
                new_fp[mutplex] = all_plex_info[mutplex]['optimize'][mutindex][0]['seq']
                new_rp[mutplex] = all_plex_info[mutplex]['optimize'][mutindex][1]['seq']
                
                new_badness = 0
                for plex_id, i in new_index.items():
                    fp = all_plex_info[plex_id]['optimize'][i][0]
                    rp = all_plex_info[plex_id]['optimize'][i][1]
                    new_badness += fp['badness'] + rp['badness']    # self badness adds on
                
                # Get the IDs of the primers that are being changed
                old_fp_id = f'fp_{mutplex}'
                old_rp_id = f'rp_{mutplex}'
                
                # Get the new primer sequences
                new_fp_seqs = new_fp[mutplex]
                new_rp_seqs = new_rp[mutplex]
                new_fp_cand, new_rp_cand = opt_cand[mutplex][mutindex]
                
                # Calculate the indices of the primers being changed
                old_fp_idx = primer_id_to_index[old_fp_id]
                old_rp_idx = primer_id_to_index[old_rp_id]
                
                # Temporarily store the old badness values
                old_fp_badness_row = inter_badness_matrix[old_fp_idx, :].copy()
                old_fp_badness_col = inter_badness_matrix[:, old_fp_idx].copy()
                old_rp_badness_row = inter_badness_matrix[old_rp_idx, :].copy()
                old_rp_badness_col = inter_badness_matrix[:, old_rp_idx].copy()
                
                # Scored out of the memo table rather than by re-folding every expansion
                # pair.
                if fix_intraplex:
                    # New primers against other plexes' current primers, plus the
                    # proposed pair itself.
                    fp_idx_of = others_idx_ns[old_fp_id]
                    rp_idx_of = others_idx_ns[old_rp_id]
                    fp_partner_cands = [curr_cand[pid]
                                        for pid in others_ids_ns[old_fp_id]]
                    rp_partner_cands = [curr_cand[pid]
                                        for pid in others_ids_ns[old_rp_id]]
                    fp_rows, rp_rows, intra_rows = pair_tbl.badness_rows(
                        [(new_fp_cand, fp_partner_cands),
                         (new_rp_cand, rp_partner_cands),
                         (new_fp_cand, [new_rp_cand])],
                        n_cpu=config['threads'])
                    intra_badness = intra_rows[0][1] if intra_rows else None
                else:
                    fp_idx_of = others_idx[old_fp_id]
                    rp_idx_of = others_idx[old_rp_id]
                    fp_partner_cands = [curr_cand[pid]
                                        for pid in others_ids[old_fp_id]]
                    rp_partner_cands = [curr_cand[pid]
                                        for pid in others_ids[old_rp_id]]
                    fp_rows, rp_rows = pair_tbl.badness_rows(
                        [(new_fp_cand, fp_partner_cands),
                         (new_rp_cand, rp_partner_cands)],
                        n_cpu=config['threads'])
                    intra_badness = None

                new_fp_interactions = [(fp_idx_of[pos], badness)
                                       for pos, badness in fp_rows]
                new_rp_interactions = [(rp_idx_of[pos], badness)
                                       for pos, badness in rp_rows]
                
                # Update the matrix with new interactions.
                for idx, badness in new_fp_interactions:
                    inter_badness_matrix[old_fp_idx, idx] = badness
                    inter_badness_matrix[idx, old_fp_idx] = badness

                for idx, badness in new_rp_interactions:
                    inter_badness_matrix[old_rp_idx, idx] = badness
                    inter_badness_matrix[idx, old_rp_idx] = badness

                if intra_badness is not None:
                    # The plex's own fP-rP dimer, for the proposed pair. Left
                    # for last, though the rows above no longer reach this cell.
                    inter_badness_matrix[old_fp_idx, old_rp_idx] = intra_badness
                    inter_badness_matrix[old_rp_idx, old_fp_idx] = intra_badness
                
                # Calculate the new inter-badness
                new_inter_badness = np.sum(inter_badness_matrix) / 2    # do we need to calculate self_badness again when updating?

                # new_comp_badness = [0,0]
                new_badness += new_inter_badness
                
                # calculate probability of accepting mutation
                if new_badness < curr_badness:
                    accept = True
                else:
                    if SATemp == 0:
                        accept = False
                    elif rand() < 2**((curr_badness - new_badness)/SATemp):
                        accept = True
                    else:
                        accept = False
                        
                # implement acceptance
                if accept:
                    curr_index[mutplex] = new_index[mutplex]
                    curr_fp[mutplex] = new_fp[mutplex]
                    curr_rp[mutplex] = new_rp[mutplex]
                    curr_badness = new_badness
                    # comp_badness = new_comp_badness

                    # Update the all_primers dictionary with the new sequences
                    all_primers[old_fp_id]['seq'] = new_fp_seqs
                    all_primers[old_rp_id]['seq'] = new_rp_seqs
                    curr_cand[old_fp_id] = new_fp_cand
                    curr_cand[old_rp_id] = new_rp_cand
                else:
                    # Revert the matrix to the old values if not accepted
                    inter_badness_matrix[old_fp_idx, :] = old_fp_badness_row
                    inter_badness_matrix[:, old_fp_idx] = old_fp_badness_col
                    inter_badness_matrix[old_rp_idx, :] = old_rp_badness_row
                    inter_badness_matrix[:, old_rp_idx] = old_rp_badness_col
                    
                learning_curve.append(curr_badness)
            # reduce SATemp
            SATemp = max(0, (SATemp - InitSATemp/NUMSTEPS))
            
        # save optimization result
        for n, (plex_id, i) in enumerate(curr_index.items()):
            all_plex_info[plex_id]['fP'] = all_plex_info[plex_id]['optimize'][i][0]
            all_plex_info[plex_id]['rP'] = all_plex_info[plex_id]['optimize'][i][1]
            # all_plex_info[plex_id]['fP_badness'] = comp_badness[0][n]
            # all_plex_info[plex_id]['rP_badness'] = comp_badness[1][n]

        all_lc.append(learning_curve)
    
    logger.info('Primer dimer optimization finished in %.3fs' % (time()-tik))
    # Candidate pairs reached and concrete folds paid.
    logger.info('pair table: %d candidate pairs filled, %d concrete pairs '
                'folded, %d pairs estimated by sampling'
                % (pair_tbl.n_filled, pair_tbl.n_concrete, pair_tbl.n_sampled))
    return all_plex_info, all_lc


def optimize_saddle(all_plex_info, config):
    '''Optimize fP and rP with stock Olivar's SADDLE heuristic
    (design.PrimerSetBadnessFast) in place of NUPACK.
    '''
    logger.info('Optimizing primer dimers (SADDLE heuristic)...')
    # set random seed
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    # automatically determine # SADDLE iterations
    pool_size = max(len(all_plex_info)/N_POOLS - 100, 0)
    InitSATemp = 1000 + 10*pool_size
    NUMSTEPS, ZEROSTEPS, TimePerStep = _sa_budget(config, pool_size)

    # generate primer pair candidates (fP-rP)
    n_pair = []
    for plex_id, plex_info in all_plex_info.items():
        optimize_list = []
        for i, fp in plex_info['fP_candidate'].iterrows():
            for j, rp in plex_info['rP_candidate'].iterrows():
                optimize_list.append([fp, rp])
        plex_info['optimize'] = optimize_list
        n_pair.append(len(optimize_list))
    logger.info('total primer pairs %d' % sum(n_pair))
    logger.info('average pairs per plex %.2f' % (sum(n_pair)/len(n_pair)))

    def pool_inter_badness(fp_dict, rp_dict):
        inter, _ = PrimerSetBadnessFast(list(fp_dict.values()), list(rp_dict.values()))
        return inter

    # optimize each tube
    all_lc = []
    tik = time()
    for i_tube in range(1, N_POOLS+1):
        logger.info('\npool %d, simulated annealing (SADDLE)...' % i_tube)
        curr_tube = [plex_id for plex_id, plex_info in all_plex_info.items() if plex_info['tube'] == i_tube]

        curr_index = {}
        curr_fp = {}
        curr_rp = {}
        curr_self_badness = 0

        for plex_id in curr_tube:
            i = floor(rand() * len(all_plex_info[plex_id]['optimize']))
            curr_index[plex_id] = i
            fp = all_plex_info[plex_id]['optimize'][i][0]
            rp = all_plex_info[plex_id]['optimize'][i][1]
            curr_fp[plex_id] = fp['seq']
            curr_rp[plex_id] = rp['seq']
            curr_self_badness += fp['badness'] + rp['badness']   # self badness adds-on

        curr_badness = curr_self_badness + pool_inter_badness(curr_fp, curr_rp)
        logger.info('initial loss = %.3f' % curr_badness)

        # optimization
        learning_curve = []
        SATemp = InitSATemp
        for step in range(NUMSTEPS + ZEROSTEPS):
            logger.info('SA temperature = %.3f, SADDLE Loss = %.3f' % (SATemp, curr_badness))
            for t in range(TimePerStep):
                # pick a plex to change primer pair
                mutplex = random.choice(curr_tube)
                n_pair_mut = len(all_plex_info[mutplex]['optimize'])
                if n_pair_mut == 1:
                    continue
                mutindex = floor(rand() * n_pair_mut)

                new_index = deepcopy(curr_index)
                new_fp = deepcopy(curr_fp)
                new_rp = deepcopy(curr_rp)

                new_index[mutplex] = mutindex
                new_fp[mutplex] = all_plex_info[mutplex]['optimize'][mutindex][0]['seq']
                new_rp[mutplex] = all_plex_info[mutplex]['optimize'][mutindex][1]['seq']

                new_self_badness = 0
                for plex_id, i in new_index.items():
                    fp = all_plex_info[plex_id]['optimize'][i][0]
                    rp = all_plex_info[plex_id]['optimize'][i][1]
                    new_self_badness += fp['badness'] + rp['badness']   # self badness adds on

                new_badness = new_self_badness + pool_inter_badness(new_fp, new_rp)

                # calculate probability of accepting mutation
                if new_badness < curr_badness:
                    accept = True
                else:
                    if SATemp == 0:
                        accept = False
                    elif rand() < 2**((curr_badness - new_badness)/SATemp):
                        accept = True
                    else:
                        accept = False

                # implement acceptance
                if accept:
                    curr_index[mutplex] = new_index[mutplex]
                    curr_fp[mutplex] = new_fp[mutplex]
                    curr_rp[mutplex] = new_rp[mutplex]
                    curr_badness = new_badness

                learning_curve.append(curr_badness)
            # reduce SATemp
            SATemp = max(0, (SATemp - InitSATemp/NUMSTEPS))

        # save optimization result
        for plex_id, i in curr_index.items():
            all_plex_info[plex_id]['fP'] = all_plex_info[plex_id]['optimize'][i][0]
            all_plex_info[plex_id]['rP'] = all_plex_info[plex_id]['optimize'][i][1]

        all_lc.append(learning_curve)

    logger.info('Primer dimer optimization (SADDLE) finished in %.3fs' % (time()-tik))
    return all_plex_info, all_lc


def to_df(all_plex_info, config):
    '''Format sequences and coordinates as a DataFrame'''
    logger.info('Formatting design output...')

    l_fP_adp = len(config['fP_prefix']) # length of adapter/prefix
    l_rP_adp = len(config['rP_prefix'])

    ref_name = []
    plex_name = []
    pool = []

    # sequence specific part of primers
    fp = []
    rp = []

    # coordinates (1-based)
    start = []
    insert_start = []
    insert_end = []
    end = []

    # sequences
    amp = []
    insert = []

    # full primer (with adapters/prefixes)
    fp_full = []
    rp_full = []

    # ARTIC columns
    art_genome = []
    art_start = []
    art_stop = []
    art_primer_name = []
    art_pool = []
    art_strand = []
    art_primer_seq = []

    for plex_id, plex_info in all_plex_info.items():
        ref_name.append(plex_info['reference'])
        plex_name.append(plex_id)
        pool.append(plex_info['tube'])
        
        curr_fp = plex_info['fP'] # fP object
        curr_rp = plex_info['rP']

        if isinstance(curr_fp['seq'], str):
            curr_fp['seq'] = [curr_fp['seq']]
        if isinstance(curr_rp['seq'], str):
            curr_rp['seq'] = [curr_rp['seq']]

        if len(curr_fp['seq']) > 1:
            max_len = max(len(seq) for seq in curr_fp['seq'])
            degenerate_seq = []
            for i in range(max_len):
                position_bases = []
                for seq in curr_fp['seq']:
                    if i < len(seq):
                        position_bases.append(seq[i])
                degenerate_base = basic.get_degenerate_base(position_bases)
                degenerate_seq.append(degenerate_base)
            curr_fp['seq'] = [''.join(degenerate_seq)]

        # Generate degenerate sequence if multiple sequences exist
        if len(curr_rp['seq']) > 1:
            # Find the maximum length among all sequences
            max_len = max(len(seq) for seq in curr_rp['seq'])
            
            degenerate_seq = []
            for i in range(max_len):
                # Get all bases at this position (for sequences that are long enough)
                position_bases = []
                for seq in curr_rp['seq']:
                    if i < len(seq):
                        position_bases.append(seq[i])
                degenerate_base = basic.get_degenerate_base(position_bases)
                degenerate_seq.append(degenerate_base)
            curr_rp['seq'] = [''.join(degenerate_seq)]

        curr_fp_seq = curr_fp['seq'][0][l_fP_adp:] # fP sequence
        curr_rp_seq = curr_rp['seq'][0][l_rP_adp:]

        fp.append(curr_fp_seq)
        rp.append(curr_rp_seq)

        # calculate coordinates
        pc = plex_info['primers_coords'] # primer design coords ([fp_start, fp_end, rp_start, rp_end])
        curr_start = pc[1] - curr_fp['dist'] - curr_fp['primer_len'] + 1
        curr_insert_start = curr_start + curr_fp['primer_len']
        curr_insert_end = pc[2] + curr_rp['dist'] - 1
        curr_end = curr_insert_end + curr_rp['primer_len']
        start.append(curr_start)
        insert_start.append(curr_insert_start)
        insert_end.append(curr_insert_end)
        end.append(curr_end)

        curr_context_seq = plex_info['context_seq']
        amp.append(curr_context_seq[curr_start-pc[0]:curr_end-pc[0]+1])
        insert.append(curr_context_seq[curr_insert_start-pc[0]:curr_insert_end-pc[0]+1])

        fp_full.append(curr_fp['seq'][0]) # fP with adapter/prefix
        rp_full.append(curr_rp['seq'][0])

        # ARTIC columns (use 0-based coordinates for BED format)
        # first fP, then rP
        art_genome.extend([plex_info['reference'], plex_info['reference']])
        art_start.extend([curr_start-1, curr_insert_end])
        art_stop.extend([curr_insert_start-1, curr_end])
        art_primer_name.extend([plex_id+'_LEFT_1', plex_id+'_RIGHT_1'])
        art_pool.extend([plex_info['tube'], plex_info['tube']])
        art_strand.extend(['+', '-'])
        art_primer_seq.extend([curr_fp_seq.upper(), curr_rp_seq.upper()])

    # format as DataFrame
    df = pd.DataFrame({
        'reference': ref_name, 
        'amplicon_id': plex_name, 
        'pool': pool, 
        'fP': fp, 
        'rP': rp, 
        'start': start, 
        'insert_start': insert_start, 
        'insert_end': insert_end, 
        'end': end, 
        'amplicon': amp, 
        'insert': insert, 
        'fP_full': fp_full, 
        'rP_full': rp_full, 
    })

    # ARTIC DataFrame
    art_df = pd.DataFrame({
        'genome': art_genome, 
        'start': art_start, 
        'stop': art_stop, 
        'primer_name': art_primer_name, 
        'pool': art_pool, 
        'strand': art_strand, 
        'primer_seq': art_primer_seq
    })
    return df, art_df
