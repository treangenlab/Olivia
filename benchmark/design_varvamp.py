#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Design tiled schemes with varVAMP.

Run in the env that has it:
    conda run -n varvamp python design_varvamp.py

One scheme per amplicon size in config.AMPLICON_SIZES, into
output/varVAMP/amp<size>/.
'''

import os
import shutil
import subprocess

import config


def design(out_dir, amp_size):
    cmd = [
        'varvamp', '--verbose', 'tiled',
        '--threads', str(config.THREADS),
        '--opt-length', str(amp_size),
        # varVAMP's --max-length is a hard cap, not a window around the
        # optimum; +10% matches the window the other two tools design in.
        '--max-length', str(int(round(amp_size * 1.1))),
        '--overlap', '25',
        # See config.VARVAMP_THRESHOLD: at its auto-estimated threshold varVAMP
        # covers a quarter of the gene with a single amplicon.
        '-t', str(config.VARVAMP_THRESHOLD),
        '-a', str(config.VARVAMP_N_AMBIG),
        config.MSA_PATH,
        out_dir,
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def main():
    for amp_size in config.AMPLICON_SIZES:
        out_dir = os.path.join(config.OUT_DIR, 'varVAMP', 'amp%d' % amp_size)
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        print('\n=== varVAMP, amplicon %d ===' % amp_size)
        design(out_dir, amp_size)


if __name__ == '__main__':
    main()
