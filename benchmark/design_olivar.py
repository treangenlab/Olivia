#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Design tiled schemes with stock Olivar 1.3.3 (SADDLE dimer scoring, no
NUPACK, no encoder filter), with the same design parameters as design_olivia.py.

Run in the env that has it:
    conda run -n olivar python design_olivar.py

Olivar builds its own .olvr rather than reusing the fork's. The two packages
pickle their reference files independently and nothing checks that they agree,
so sharing one would risk comparing designs made against different references
while looking like it had controlled for exactly that.
'''

import os
import shutil

from olivar import build, tiling

import config


def build_reference(work_dir):
    os.makedirs(work_dir, exist_ok=True)
    # The .olvr is named after the MSA file.
    stem = os.path.splitext(os.path.basename(config.MSA_PATH))[0]
    ref_path = os.path.join(work_dir, '%s.olvr' % stem)
    if os.path.isfile(ref_path):
        print('reusing %s' % ref_path)
        return ref_path
    build(
        fasta_path=None,
        msa_path=config.MSA_PATH,
        var_path=None,
        BLAST_db=None,
        out_path=work_dir,
        title=None,
        threads=config.THREADS,
        align=False,
        min_var=0.01,
        deg=False,
    )
    if not os.path.isfile(ref_path):
        raise FileNotFoundError('build() did not produce %s' % ref_path)
    return ref_path


def main():
    work_dir = os.path.join(config.OUT_DIR, 'olivar_work')
    ref_path = build_reference(work_dir)

    for amp_size in config.AMPLICON_SIZES:
        out_dir = os.path.join(config.OUT_DIR, 'Olivar', 'amp%d' % amp_size)
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        print('\n=== Olivar, amplicon %d ===' % amp_size)
        # Every argument matched to design_olivia.py; stock Olivar's tiling()
        # takes the same signature minus the encoder parameters.
        tiling(
            ref_path=ref_path,
            out_path=out_dir,
            title='design',

            max_amp_len=int(round(amp_size * 1.1)),
            min_amp_len=max(120, int(round(amp_size * 0.9))),

            w_egc=1.0, w_lc=1.0, w_ns=1.0, w_var=1.0, w_sensi=1.0, w_combi=1.0,

            temperature=60.0,
            salinity=0.18,
            dG_max=-11.8,
            min_GC=0.2,
            max_GC=0.75,
            min_complexity=0.4,
            max_len=36,
            check_var=False,

            fP_prefix='', rP_prefix='',

            seed=config.SEED,
            threads=config.THREADS,
            iterMul=1,
            deg=False,
        )


if __name__ == '__main__':
    main()
