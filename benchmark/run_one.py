#!/usr/bin/env python3
'''
Run one tool's design step on one target; rerun_all.py calls this per target.

    python run_one.py TOOL MSA_PATH OUT_DIR THREADS PS3_FREQ VV_THRESH VV_AMBIG

Patches config in place before the design script reads it, so the per-target
settings each original run used are reproduced without editing config.py.
VV_THRESH 'auto' leaves -t off and lets varVAMP pick it (SARS-CoV-2 run).
'''

import sys

import config

tool, msa, out_dir, threads, ps3_freq, vv_thresh, vv_ambig = sys.argv[1:8]
config.MSA_PATH = msa
config.OUT_DIR = out_dir
config.THREADS = int(threads)
config.PS3_MIN_BASE_FREQ = float(ps3_freq)
config.VARVAMP_THRESHOLD = vv_thresh
config.VARVAMP_N_AMBIG = int(vv_ambig)

if tool == 'Olivia':
    import design_olivia as m
elif tool == 'Olivar':
    import design_olivar as m
elif tool == 'PrimalScheme3':
    import design_primalscheme3 as m
elif tool == 'varVAMP':
    import design_varvamp as m
    if vv_thresh == 'auto':
        _run = m.subprocess.run

        def run(cmd, **kw):
            i = cmd.index('-t')
            return _run(cmd[:i] + cmd[i + 2:], **kw)
        m.subprocess.run = run
else:
    sys.exit('unknown tool %s' % tool)

m.main()
