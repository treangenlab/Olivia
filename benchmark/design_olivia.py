#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Design a tiled scheme with Olivia in non-degenerate mode (Olivia-nondeg).

    python design_olivia.py
'''

import os
import sys
import shutil

import config

sys.path.insert(0, config.SRC_DIR)
from olivia import build, tiling


def build_reference(work_dir, msa_path=None, deg=False, cons_thres=0.7):
    '''The .olvr reference the scheme is tiled against, built once per work_dir and reused.
    '''
    msa_path = msa_path or config.MSA_PATH
    os.makedirs(work_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(msa_path))[0]
    ref_path = os.path.join(work_dir, '%s.olvr' % stem)
    if os.path.isfile(ref_path):
        print('reusing %s' % ref_path)
        return ref_path
    build(
        fasta_path=None,
        msa_path=msa_path,
        var_path=None,
        BLAST_db=None,
        out_path=work_dir,
        title=None,
        threads=config.THREADS,
        align=False,      # the input is already aligned
        min_var=0.01,
        deg=deg,
        cons_thres=cons_thres,
    )
    if not os.path.isfile(ref_path):
        raise FileNotFoundError('build() did not produce %s' % ref_path)
    return ref_path


def design(ref_path, out_dir, title, amp_size, keep_frac, deg=False,
           primer_gate=None, pair_scorer='nupack', extra=None):
    tiling(
        ref_path=ref_path,
        out_path=out_dir,
        title=title,

        # Same +/-10% amplicon window PrimalScheme3 uses.
        max_amp_len=int(round(amp_size * 1.1)),
        min_amp_len=max(120, int(round(amp_size * 0.9))),

        # w_sensi and w_combi are inert at deg=False and carry the mismatch
        # and permutation penalties at deg=True.
        w_egc=1.0, w_lc=1.0, w_ns=1.0, w_var=1.0, w_sensi=1.0, w_combi=1.0,

        temperature=60.0,
        salinity=0.18,
        dG_max=-11.8,
        min_GC=0.2,
        max_GC=0.75,
        min_complexity=0.4,
        max_len=30,
        check_var=False,

        fP_prefix='', rP_prefix='',

        seed=config.SEED,
        threads=config.THREADS,
        iterMul=1,
        deg=deg,

        encoder_keep_frac=keep_frac,
        encoder_model=config.ENCODER_MODEL,
        pair_scorer=pair_scorer,
        primer_gate=(config.OLIVIA_PRIMER_GATE if primer_gate is None
                     else primer_gate),
        max_deg=config.OLIVIA_MAX_DEG,

        **dict(config.OLIVIA_TILING_EXTRA, **(extra or {})),
    )


def main():
    work_dir = os.path.join(config.OUT_DIR, 'olivia_work')
    ref_path = build_reference(work_dir)

    for amp_size in config.AMPLICON_SIZES:
        out_dir = os.path.join(config.OUT_DIR, 'Olivia', 'amp%d' % amp_size)
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        print('\n=== Olivia, amplicon %d ===' % amp_size)
        design(ref_path, out_dir, 'design', amp_size, config.ENCODER_KEEP_FRAC)


if __name__ == '__main__':
    main()
