#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''Main workflow of Olivia: build(), tiling(), save(), specificity(), sensitivity().'''


__author__ = 'Michael X. Wang'


import os
CURRENT_DIR = os.path.dirname(__file__)

import sys
import io
import json
import pickle # needs python>=3.8

import logging
import time

# Single-threaded BLAS in the parent; parallelism comes from the worker pool.
for _thread_var in (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
):
    os.environ.setdefault(_thread_var, '1')

# logging settings
logFormatter = logging.Formatter(
    fmt='%(asctime)s | %(levelname)-8s | %(message)s', 
    style='%', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
# add handler for printing to stdout
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
# create logger
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG) # show all messages
logger.addHandler(consoleHandler)

import pandas as pd
import numpy as np

# internal modules
import basic
# Aliased: tiling() has a parameter named primer_gate.
import primer_gate as olivar_primer_gate
from ncbi_tools import ns_simulation
# from design import PrimerSetBadnessFast
from msa_tools import run_validate, run_preprocess, run_cmd, DEFAULT_CONS_THRES

# helper modules
from build_helper import run_build
from tiling_helper import design_context_seq, get_primer, optimize, optimize_saddle, to_df
from encoder_filter import filter_primer_candidates
from save_helper import save
from msa_tools import expand_degenerate_sequence, average_scores

from nupack_helper import calculate_primer_pair, mp_wrapper
import nupack_helper


REFEXT = '.olvr' # extension for Olivar reference file
DESIGNEXT = '.olvd' # extension for Olivar design file

def build(fasta_path: str, msa_path: str, var_path: str, BLAST_db: str, out_path: str,
    title: str, threads: int, align: bool, min_var: float, deg: bool,
    cons_thres: float = DEFAULT_CONS_THRES):
    '''
    Build the Olivar reference file for tiled amplicon design, handling gaps in the MSA.
    Input:
        fasta_path: Optional, Path to the FASTA reference sequence. The sequence should be high-quality and contain no degenerate bases
        msa_path: Optional, Path to the MSA file (Multiple Sequence Alignment in FASTA format).
        var_path: Optional, path to the CSV file of SNP coordinates and frequencies. 
            Required columns: "START", "STOP", "FREQ". "FREQ" is considered as 1.0 if empty. Coordinates are 1-based.
        BLAST_db: Optional, path to the BLAST database. 
        out_path: Output directory [./]. 
        title: Name of the Olivar reference file [MSA record ID]. 
        threads: Number of threads [1]. 
        align: Control whether do alignment for MSA file or not [False]. 
        min_var: Minimum frequency threshold for sequence variations generated from the input MSA [0.01].
        deg: Control whether use degenerate mode or not. This mode only works with MSA input.
        cons_thres: Accumulation threshold of the degenerate consensus [0.7]. A column stays
            non-degenerate whenever its most frequent base already reaches this, so the default
            ignores any variant below 30% -- which is most of the variation on a conserved
            target. Raising it buys sequence coverage and costs permutations; permutations
            multiply along a primer, so tiling()'s max_deg is what keeps the cost bounded.
            Inert at deg=False.
    '''
    if not msa_path and not fasta_path:
        raise ValueError("Either 'msa_path' or 'fasta_path' must be provided.")
    if msa_path:
        if fasta_path:
            logger.warning("Both 'msa_path' and 'fasta_path' provided. Ignoring 'fasta_path' and processing with MSA + BLAST.")
        if not os.path.exists(msa_path):
            raise FileNotFoundError(f"MSA file '{msa_path}' not found.")
        fasta_path = None
        msa_filename = os.path.splitext(os.path.basename(msa_path))[0]
        if align:
            logger.info("Running alignment with MAFFT...")
            aligned_msa_path = os.path.join(out_path, os.path.basename(msa_path).replace('.fasta', '_aligned.fasta'))
            msa_str = run_cmd('mafft', '--auto', '--thread', str(threads), msa_path)

            if not os.path.exists(out_path):
                os.makedirs(out_path)
            with open(aligned_msa_path, 'w') as f:
                f.write(msa_str)
            msa_path = aligned_msa_path
        #fasta_path, var_path = run_preprocess(io.StringIO(msa_str), msa_filename, out_path, threads)
        fasta_path, var_path = run_preprocess(msa_path, msa_filename, out_path, threads, min_var, deg, cons_thres)
    else:
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA file '{fasta_path}' not found.")
        if deg: 
            logger.warning('Degenerate mode only supported by MSA input. Will proceed in normal mode...')
            deg = False
    if deg:
        run_build(fasta_path, msa_path, var_path, BLAST_db, out_path, title, threads, deg=deg,
                  cons_thres=cons_thres)
    else:
        run_build(fasta_path, None, var_path, BLAST_db, out_path, title, threads, deg=deg)


def tiling(ref_path: str, out_path: str, title: str, max_amp_len: int, min_amp_len: int, 
    w_egc: float, w_lc: float, w_ns: float, w_var: float, w_sensi: float, w_combi: float,
    temperature: float, salinity: float, dG_max: float, min_GC: float, max_GC: float, 
    min_complexity: float, max_len: int, check_var: bool, fP_prefix: str, rP_prefix: str,
    seed: int, threads: int, iterMul: int, deg: bool,
    encoder_keep_frac: float = 0.5, encoder_model: str = None, pair_scorer: str = 'nupack',
    primer_gate = 'lenient', max_deg: int = 128,
    num_steps: int = None, zero_steps: int = None, time_per_step: int = None,
    canonical_folding: bool = True, fix_intraplex: bool = True,
    deg_sample: int = 0, precompute_pairs: bool = True,
    precompute_batch: int = 2000000):
    '''Design tiled amplicons.
    Input:
        ref_path: Path to the Olivar reference file (.olvr), or a directory of
        reference files.
        out_path: Output directory [./].
        title: Name of this design [olivar-design].
        max_amp_len: Maximum amplicon length [420].
        min_amp_len: Minimum amplicon length [0.9*max_amp_len, at least 120].
        w_egc, w_lc, w_ns, w_var: Weights for extreme GC, low complexity,
        non-specificity, variations [1.0].
        w_sensi, w_combi: Weights for MSA sensitivity and log2 permutation count
        (deg mode) [1.0].
        temperature: PCR annealing temperature [60.0].
        salinity: Concentration of monovalent ions in molar [0.18].
        dG_max: Maximum free energy change of a primer in kcal/mol [-11.8].
        min_GC, max_GC: GC content limits of a primer [0.2, 0.75].
        min_complexity: Minimum sequence complexity of a primer [0.4].
        max_len: Maximum length of a primer [36].
        check_var: Drop candidates with variations within 5 nt of the 3' end
        [False].
        fP_prefix, rP_prefix: Forward / reverse primer prefixes [''].
        seed: Random seed [10].
        threads: Number of threads [1].
        iterMul: Multiplier of iterations during PDR optimization [1].
        deg: Design degenerate primers.
        encoder_keep_frac: Fraction of candidates the encoder keeps per plex and
        direction [0.5; 1.0 = off].
        encoder_model: Encoder checkpoint [results/encoder/encoder.pt or
        $OLIVIA_MODEL].
        pair_scorer: Dimer scorer for pool optimization: 'nupack' or 'saddle'
        (Olivar's heuristic).
        primer_gate: Hard candidate limits: 'lenient' (Olivar's) or 'varvamp'
        (varVAMP defaults), or a dict.
        max_deg: Reject candidates with more expansions than this [128]; raised 4x
        per relaxation round.
        num_steps, zero_steps, time_per_step: Simulated-annealing budget [derived
        from pool size].
        canonical_folding: Fold strand pairs in a canonical orientation [True].
        fix_intraplex: Score a plex's own fP-rP dimer for the proposed pair [True].
        deg_sample: Expansion pairs folded per degenerate candidate pair [0 = all,
        exact; paper: 64].
        precompute_pairs: Fill the pair deltaG table in batches before annealing
        [True].
        precompute_batch: Concrete pairs per precompute batch; lower it to reduce
        memory [2000000].
    '''
    start_time = time.time()
    if pair_scorer not in ('nupack', 'saddle'):
        raise ValueError('pair_scorer must be "nupack" or "saddle", got %r' % pair_scorer)
    # Resolved here purely to fail fast on a typo or a missing primer3: the gate
    # is otherwise not touched until get_primer(), well over a minute in.
    olivar_primer_gate.PrimerGate.from_spec(primer_gate)
    if deg:
        logger.info('Designing in degenerate mode...')
    else:
        logger.info('Designing in normal mode...')
    
    config = {
        'ref_path': ref_path, 
        'out_path': out_path, 
        'title': title, 
        'max_amp_len': max_amp_len, 
        'min_amp_len': min_amp_len, 
        'w_egc': w_egc, 
        'w_lc': w_lc, 
        'w_ns': w_ns, 
        'w_var': w_var, 
        
        'w_sensi': w_sensi,
        'w_combi': w_combi,
        
        'temperature': temperature, 
        'salinity': salinity, 
        'dG_max': dG_max, 
        'min_GC': min_GC, 
        'max_GC': max_GC, 
        'min_complexity': min_complexity, 
        'max_len': max_len, 
        'check_SNP': check_var, 
        'fP_prefix': fP_prefix, 
        'rP_prefix': rP_prefix, 
        'seed': seed, 
        'threads': threads,
        'iterMul': iterMul,
        'encoder_keep_frac': encoder_keep_frac,
        'encoder_model': encoder_model,
        'pair_scorer': pair_scorer,
        'primer_gate': primer_gate,
        'max_deg': max_deg,
        # Simulated-annealing budget.
        'NumSteps': num_steps,
        'ZeroSteps': zero_steps,
        'TimePerStep': time_per_step,
        'canonical_folding': canonical_folding,
        'fix_intraplex': fix_intraplex,
        # Expansion pairs folded per candidate pair (0 = all, exact; see pair_table).
        'deg_sample': deg_sample,
        # Fill the candidate-pair table up front in large batches instead of lazily per
        # trial.
        'precompute_pairs': precompute_pairs,
        'precompute_batch': precompute_batch
    }

    if not os.path.exists(config['out_path']):
        os.makedirs(config['out_path'])

    # store config values
    design_ref_path = config['ref_path']

    # validate input .olvr file(s)
    ref_path_dict = dict() # {ref_name: ref_path}


    # log to file, must happen after working_dir is created
    # if working_dir exist, it should be a directory as well
    fileHandler = logging.FileHandler(os.path.join(out_path, f'{config['title']}.log'), mode='a')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    # check input is a file or a directory
    if os.path.isfile(design_ref_path) and design_ref_path.endswith(REFEXT):
        # use the name of the .olvr file as reference name
        file = os.path.basename(design_ref_path)
        ref_path_dict[file[:-len(REFEXT)]] = design_ref_path
    elif os.path.isdir(design_ref_path):
        for file in sorted(os.listdir(design_ref_path)):
            file_path = os.path.join(design_ref_path, file)
            if os.path.isfile(file_path) and file_path.endswith(REFEXT):
                # use file name as reference name
                ref_path_dict[file[:-len(REFEXT)]] = file_path
        if ref_path_dict:
            logger.info(f'Found {len(ref_path_dict)} {REFEXT} file(s) under "{design_ref_path}"')
            for file_path in ref_path_dict.values():
                logger.info(os.path.basename(file_path))
        else:
            raise FileNotFoundError(f'No {REFEXT} file found in the directory "{design_ref_path}".')
    else:
        raise FileNotFoundError(f'Input is neither a {REFEXT} file nor a directory.')
    
    # design PDRs for each reference
    all_plex_info = {}
    all_ref_info = {}
    for ref_name, ref_path in ref_path_dict.items():
        config['ref_path'] = ref_path
        logger.info(f'Designing PDRs for {ref_name} using {ref_path}...')
        temp_dict, risk_arr, gc_arr, comp_arr, hits_arr, var_arr, sensi_arr, combi_arr, all_loss, seq_record = design_context_seq(config, deg=deg)
        all_plex_info.update(temp_dict)
        all_ref_info[ref_name] = {
            'risk_arr': risk_arr, 
            'gc_arr': gc_arr, 
            'comp_arr': comp_arr, 
            'hits_arr': hits_arr, 
            'var_arr': var_arr, 

            'sensi_arr': sensi_arr,
            'combi_arr': combi_arr,

            'all_loss': all_loss, 
            'seq_record': seq_record, 
        }
        """for i, sensi_value in enumerate(sensi_arr):
            logger.debug(f"  Element {i}: {sensi_value}")"""

    
    # revert modified config values
    config['ref_path'] = design_ref_path
    # Set before get_primer(): candidate self-badness also uses the deltaG pool.
    nupack_helper.set_n_cpu(config['threads'])
    # Before any folding happens, and in the parent only -- workers are handed
    # pairs that have already been canonicalized.
    nupack_helper.set_canonical_folding(config.get('canonical_folding'))
    if config.get('canonical_folding'):
        logger.info('canonical strand orientation: on')
    all_plex_info_primer = get_primer(all_plex_info, config)
    all_plex_info_primer = filter_primer_candidates(all_plex_info_primer, config)
    optimize_fn = optimize_saddle if config['pair_scorer'] == 'saddle' else optimize
    all_plex_info_optimize, learning_curve = optimize_fn(all_plex_info_primer, config)
    df, art_df = to_df(all_plex_info_optimize, config)

    # format and save design output
    design_out = {
        'config': config, 
        'df': df, 
        'art_df': art_df, # ARTIC output format
        'all_plex_info': all_plex_info_optimize, 
        'learning_curve': learning_curve, 
        'all_ref_info': all_ref_info
    }
    design_path = os.path.join(config['out_path'], '%s.olvd' % config['title'])
    with open(design_path, 'wb') as f:
        pickle.dump(design_out, f, protocol=5) # protocol 5 needs python>=3.8
        logger.info('Design file saved as %s' % design_path)

    # save human readable files
    save(design_out, config['out_path'])

    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Tiling took {total_time:.4f} seconds to execute")


def specificity(primer_pool: str, pool: int, BLAST_db: str, out_path: str, 
    title: str, max_amp_len: int, temperature: float, threads: int):
    '''
    Run analysis on a primer pool for multiplex PCR
    Input:
        primer_pool: Path to the csv file of a primer pool. 
            Required columns: "amplicon_id" (amplicon name), "fP" (sequence of forward primer), "rP" (sequence of reverse primer), "pool" (pool number, e.g., 1).
        pool: Primer pool number [1]. 
        BLAST_db: Optional, path to the BLAST database. 
            Note that this path should end with the name of the BLAST database (e.g., "example_input/Human/GRCh38_primary").
        out_path: Output directory [./]. 
        title: Name of validation [olivar-val].
        max_amp_len: Maximum length of predicted non-specific amplicon [1500]. 
        temperature: PCR annealing temperature [60.0]. 
        threads: Number of threads [1]. 
    '''
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    seq_list = []
    seq_names = []
    df = pd.read_csv(primer_pool, sep=',', index_col=False)

    # column name mapper for backward compatibility
    df.rename(columns={
        'amp_id': 'amplicon_id', 
        'amp': 'amplicon'
    }, inplace=True)
    
    for i, row in df[df['pool']==pool].iterrows():
        seq_list.extend([row['fP'], row['rP']])
        seq_names.extend(['%s_fP' % row['amplicon_id'], '%s_rP' % row['amplicon_id']])
    logger.info(f'Successfully loaded {primer_pool}, pool-{pool}.')

    # expand all sequences
    expanded_seqs = []      # store all primers, contain duplicates
    expanded_names = []
    for i, seq in enumerate(seq_list):
        expanded = expand_degenerate_sequence(seq)
        for j, var_seq in enumerate(expanded):
            expanded_seqs.append(var_seq)
            expanded_names.append(f"{seq_names[i]}_var{j}")

    # non-specific simulation
    if BLAST_db:
        # Run BLAST on all expanded sequences
        df_ns_expanded, df_count_expanded, all_expanded_hits = ns_simulation(
            BLAST_db, expanded_seqs, expanded_names, max_amp_len, threads
        )

        name_to_seq = dict(zip(expanded_names, expanded_seqs))
        # --- Non-specific amplicons ---
        df_ns = df_ns_expanded.copy()
        df_ns["fP_seq"] = df_ns["fP"].map(name_to_seq)
        df_ns["rP_seq"] = df_ns["rP"].map(name_to_seq)
        save_path = os.path.join(out_path, f'{title}_pool-{pool}_ns-amp.csv')
        df_ns.to_csv(save_path, index=False)
        logger.info('Non-specific amplicons saved as %s' % save_path)

        # --- Non-specific primer pairs ---
        df_count = df_count_expanded.copy()
        df_count["fP_seq"] = df_count["fP"].map(name_to_seq)
        df_count["rP_seq"] = df_count["rP"].map(name_to_seq)
        save_path = os.path.join(out_path, f'{title}_pool-{pool}_ns-pair.csv')
        df_count.to_csv(save_path, index=False)
        logger.info('Non-specific primer pairs saved as %s' % save_path)

        all_hits = all_expanded_hits[:]  # keep each variant's hit count
    else:
        all_hits = np.zeros(len(seq_list)) - 1
        logger.info('No BLAST database provided, skipped.')
    expanded_seq_list = expanded_seqs
    expanded_seq_names = expanded_names

    # calculate Badness for each expanded sequence
    all_bad = [0.0] * len(expanded_seq_list)
    self_bad = [0.0] * len(expanded_seq_list)
    all_pairs = []
    self_pairs = []
    
    # Expand each sequence once rather than once per pair.
    n_exp = len(expanded_seq_list)
    _PAIR_BLOCK = 200000

    def _key(a, b):
        return (a, b) if a <= b else (b, a)

    def _flush(block):
        '''block is [(i, j, key)]; add each pair's badness to all_bad.'''
        if not block:
            return
        dG_of = nupack_helper._batched_dG(
            list({k for _, _, k in block}), temperature, n_cpu=threads)
        for i, j, k in block:
            dG = dG_of.get(k)
            if dG is None:
                continue
            badness = max(0, -dG)
            all_bad[i] += badness
            all_bad[j] += badness

    block = []
    for i in range(n_exp):
        seq_i = expanded_seq_list[i]
        for j in range(i + 1, n_exp):
            block.append((i, j, _key(seq_i, expanded_seq_list[j])))
            if len(block) >= _PAIR_BLOCK:
                _flush(block)
                block = []
    _flush(block)

    self_keys = [_key(s, s) for s in expanded_seq_list]
    self_dG = nupack_helper._batched_dG(list(set(self_keys)), temperature,
                                        n_cpu=threads)
    for i, k in enumerate(self_keys):
        dG = self_dG.get(k)
        if dG is not None:
            self_bad[i] += max(0, -dG)

    # calculate dG
    gen = nupack_helper.primer_generator(temperature=temperature, salinity=0.18)
    all_dG = average_scores(expanded_seq_list, lambda p: gen.dG_init + gen.StacksDG(p))

    df_val = pd.DataFrame({
        'name': expanded_seq_names,
        'seq': expanded_seq_list,
        'length': [len(p) for p in expanded_seq_list],
        '%GC': average_scores(expanded_seq_list, basic.get_GC),
        'complexity': average_scores(expanded_seq_list, basic.get_complexity),
        f'dG ({temperature}C)': all_dG,
        'self_align': self_bad,
        'dimer_score': all_bad,
        'BLAST_hits': all_hits
    })
    save_path = os.path.join(out_path, f'{title}_pool-{pool}.csv')
    df_val.to_csv(save_path, index=False)
    logger.info('Validation file saved as %s' % save_path)

def sensitivity(primer_pool: str, msa_path: str, pool: int, out_path: str, 
    title: str, temperature: float=60.0, sodium: float=0.18, threads: int=1, align: bool=False):
    '''
    Visualize an MSA, along with its primers/probes (if provided), and validate their alignment and sensitivity.
    Input:
        primer_pool: Path to the csv file of a primer pool. 
            Required columns: "amplicon_id" (amplicon name), "fP" (sequence of forward primer), "rP" (sequence of reverse primer), "pool" (pool number, e.g., 1).
        msa_path: Path to the MSA file.
        pool: Primer pool number [1]. 
        out_path: Output directory [./]. 
        title: Name of validation [olivar-val].
        temperature: PCR annealing temperature [60.0]. 
        sodium: The sum of the concentrations of monovalent ions (Na+, K+, NH4+), in molar [0.18].
        threads: Number of threads [1]. 
        align: Conrol whether do alignment for MSA file or not. [False]. 
    '''
    if not msa_path:
        raise ValueError("'msa_path' must be provided for sensitivity calculation.")
    
    if align:
        logger.info("Running alignment with MAFFT...")
        aligned_msa_path = os.path.join(out_path, os.path.basename(msa_path).replace('.fasta', '_aligned.fasta'))
        msa_str = run_cmd('mafft', '--auto', '--thread', str(threads), msa_path)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with open(aligned_msa_path, 'w') as f:
            f.write(msa_str)
        msa_path = aligned_msa_path
    
    run_validate(msa_path = msa_path, primer_pool = primer_pool, pool = pool, temperature = temperature, sodium = sodium, out_path = out_path, title = title, n_cpu = threads)
