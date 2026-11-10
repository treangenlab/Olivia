#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Design tiled schemes with PrimalScheme3.

Run in the env that has it:
    conda run -n primalscheme3 python design_primalscheme3.py

One scheme per amplicon size in config.AMPLICON_SIZES, into
output/PrimalScheme3/amp<size>/.
'''

import os
import re
import shutil
import subprocess
import sys

import config

# primalscheme3 needs a simple first header; sanitize a scratch copy of the MSA.
CHROM_REGEX = re.compile(r'[^a-zA-Z0-9_.]')


def sanitize_msa(src_path, dst_path):
    with open(src_path) as fin, open(dst_path, 'w') as fout:
        for line in fin:
            if line.startswith('>'):
                header, sep, rest = line[1:].rstrip('\n').partition(' ')
                header = CHROM_REGEX.sub('_', header)
                fout.write('>' + header + sep + rest + '\n')
            else:
                fout.write(line)


def design(msa_path, out_dir, amp_size):
    cmd = [
        'primalscheme3', 'scheme-create',
        '--msa', msa_path,
        '--output', out_dir,
        '--amplicon-size', str(amp_size),
        '--n-pools', str(config.N_POOLS),
        # See config.PS3_MIN_BASE_FREQ: at the default 0.0 this alignment
        # digests to zero valid kmers and the run aborts.
        '--min-base-freq', str(config.PS3_MIN_BASE_FREQ),
        '--ignore-n',
        '--ncores', str(config.THREADS),
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def main():
    tool_dir = os.path.join(config.OUT_DIR, 'PrimalScheme3')
    os.makedirs(tool_dir, exist_ok=True)
    sanitized_msa = os.path.join(tool_dir, 'sanitized_msa.fasta')
    sanitize_msa(config.MSA_PATH, sanitized_msa)

    for amp_size in config.AMPLICON_SIZES:
        out_dir = os.path.join(tool_dir, 'amp%d' % amp_size)
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        print('\n=== PrimalScheme3, amplicon %d ===' % amp_size)
        design(sanitized_msa, out_dir, amp_size)


if __name__ == '__main__':
    main()
