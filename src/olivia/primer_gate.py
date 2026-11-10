#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Optional varVAMP-style hard limits on primer candidates (primer_gate='varvamp').

Adds varVAMP's published defaults -- Tm 56-63 degC, GC 35-65%, polyX <= 4,
dinucleotide repeats <= 4, 1-3 G+C in the last 5 nt, hairpin Tm <= 47 degC,
homodimer Tm <= 35 degC and dG >= -9 kcal/mol -- to Olivar's GC, complexity and
dG rules. Length is not constrained.
'''


import logging
logger = logging.getLogger('main')

from msa_tools import expand_degenerate_sequence


# varVAMP's PCR conditions (default_config.py, PCR_*). primer3 takes these per
# call rather than from global state, so they travel with every calculation.
PCR_CONDITIONS = {
    'mv_conc': 100.0,    # monovalent cations, mM
    'dv_conc': 2.0,      # divalent cations, mM
    'dntp_conc': 0.8,    # dNTPs, mM
    'dna_conc': 15.0,    # primer, nM
}

# Degenerate candidates must pass on every expansion; at most this many are checked.
MAX_PERMUTATIONS = 64

VARVAMP_GATE = {
    # Applied through Olivar's own min_GC/max_GC (see olivar_overrides).
    'gc': (35.0, 65.0),          # percent, converted to Olivar's fraction

    # No 'length' key: see the docstring. Olivar's max_len stays wherever the
    # caller set it, so the dG growth rule can finish.

    # Axes Olivar has no parameter for. These are the gate proper.
    'tm': (56.0, 63.0),
    'max_polyx': 4,
    'max_dinuc_repeats': 4,
    'gc_clamp_3p': (1, 3),       # G+C count in the last 5 nt
    'hairpin_tm_max': 47.0,
    'homodimer_tm_max': 35.0,
    'homodimer_dg_min': -9000.0, # cal/mol, primer3's unit
}

GATE_PRESETS = {
    'lenient': None,
    'varvamp': VARVAMP_GATE,
}

# Fail keys in test order; get_primer() relaxes on each.
GATE_FAIL_KEYS = (
    'polyx_fail',
    'dinuc_fail',
    'gc_clamp_fail',
    'tm_fail',
    'hairpin_fail',
    'homodimer_fail',
)


def calc_max_polyx(seq):
    '''Longest homopolymer run. varVAMP primers.py calc_max_polyx.'''
    previous_nuc, counter, max_polyx = seq[0], 1, 1
    for nuc in seq[1:]:
        if nuc == previous_nuc:
            counter += 1
        else:
            counter = 1
            previous_nuc = nuc
        if counter > max_polyx:
            max_polyx = counter
    return max_polyx


def calc_max_dinuc_repeats(seq):
    '''Longest dinucleotide repeat.'''
    max_dinuc = 1
    for s in [seq, seq[1:]]:
        previous_dinuc = s[0:2]
        max_dinuc = 1
        counter = 1
        for i in range(2, len(s), 2):
            if s[i:i+2] == previous_dinuc:
                counter += 1
            else:
                if counter > max_dinuc:
                    max_dinuc = counter
                counter = 1
                previous_dinuc = s[i:i+2]
    return max_dinuc


def calc_end_gc(seq):
    '''G+C count in the last 5 bases of the 3' end. varVAMP primers.py.'''
    return seq[-5:].count('g') + seq[-5:].count('c')


class PrimerGate(object):
    '''A resolved set of per-candidate limits.'''

    def __init__(self, spec):
        self.spec = dict(spec)
        # primer3 is imported here rather than at module scope: the lenient path
        # is stock Olivar and must not acquire a new hard dependency.
        import primer3
        self._p3 = primer3

    @classmethod
    def from_spec(cls, spec):
        '''Resolve a preset name, an explicit dict, or None into a gate or None.'''
        if spec is None:
            return None
        if isinstance(spec, str):
            if spec not in GATE_PRESETS:
                raise ValueError(
                    'primer_gate must be one of %s or a dict of overrides, got %r'
                    % (sorted(GATE_PRESETS), spec))
            resolved = GATE_PRESETS[spec]
            if resolved is None:
                return None
            return cls(resolved)
        if isinstance(spec, dict):
            unknown = set(spec) - set(VARVAMP_GATE)
            if unknown:
                raise ValueError('unknown primer_gate key(s): %s. Known keys: %s'
                                 % (sorted(unknown), sorted(VARVAMP_GATE)))
            merged = dict(VARVAMP_GATE)
            merged.update(spec)
            return cls(merged)
        raise TypeError('primer_gate must be a preset name, a dict, or None, got %r'
                        % type(spec).__name__)

    def olivar_overrides(self):
        '''The subset of the gate that Olivar already enforces, in Olivar's units.'''
        gc_min, gc_max = self.spec['gc']
        return {
            'min_GC': gc_min / 100.0,
            'max_GC': gc_max / 100.0,
        }

    def _permutations(self, seq):
        # Lowercase first: Olivar marks SNPs in uppercase, and the varVAMP helpers are
        # case-sensitive.
        expansions = expand_degenerate_sequence(seq.lower())
        return expansions[:MAX_PERMUTATIONS]

    def check(self, seq):
        '''None if the candidate passes, otherwise the GATE_FAIL_KEYS entry it failed on
        first.
        '''
        for s in self._permutations(seq):
            fail = self._check_one(s)
            if fail is not None:
                return fail
        return None

    def _check_one(self, s):
        if calc_max_polyx(s) > self.spec['max_polyx']:
            return 'polyx_fail'
        if calc_max_dinuc_repeats(s) > self.spec['max_dinuc_repeats']:
            return 'dinuc_fail'
        clamp_min, clamp_max = self.spec['gc_clamp_3p']
        if not clamp_min <= calc_end_gc(s) <= clamp_max:
            return 'gc_clamp_fail'

        upper = s.upper()
        tm_min, tm_max = self.spec['tm']
        if not tm_min <= self._p3.calc_tm(upper, **PCR_CONDITIONS) <= tm_max:
            return 'tm_fail'
        if self._p3.calc_hairpin(upper, **PCR_CONDITIONS).tm > self.spec['hairpin_tm_max']:
            return 'hairpin_fail'

        homodimer = self._p3.calc_homodimer(upper, **PCR_CONDITIONS)
        if (homodimer.tm > self.spec['homodimer_tm_max']
                or homodimer.dg < self.spec['homodimer_dg_min']):
            return 'homodimer_fail'
        return None

    def relaxed(self, fail_mode, step=1):
        '''Copy of this gate with the `fail_mode` limit widened one step, or None if it
        cannot widen.
        '''
        spec = dict(self.spec)
        if fail_mode == 'tm_fail':
            lo, hi = spec['tm']
            spec['tm'] = (lo - step, hi + step)
        elif fail_mode == 'polyx_fail':
            spec['max_polyx'] += step
        elif fail_mode == 'dinuc_fail':
            spec['max_dinuc_repeats'] += step
        elif fail_mode == 'gc_clamp_fail':
            lo, hi = spec['gc_clamp_3p']
            spec['gc_clamp_3p'] = (max(0, lo - step), min(5, hi + step))
            if spec['gc_clamp_3p'] == (0, 5) and self.spec['gc_clamp_3p'] == (0, 5):
                return None      # already vacuous, widening it again changes nothing
        elif fail_mode == 'hairpin_fail':
            spec['hairpin_tm_max'] += 2 * step
        elif fail_mode == 'homodimer_fail':
            spec['homodimer_tm_max'] += 2 * step
            spec['homodimer_dg_min'] -= 500.0 * step
        else:
            return None
        return PrimerGate(spec)

    def describe(self):
        '''One-line summary for the design log.'''
        return ('Tm %g-%g, GC %g-%g%%, polyX<=%d, dinuc<=%d, '
                '3\' clamp %d-%d, hairpin Tm<=%g, homodimer Tm<=%g/dG>=%g'
                % (self.spec['tm'][0], self.spec['tm'][1],
                   self.spec['gc'][0], self.spec['gc'][1],
                   self.spec['max_polyx'], self.spec['max_dinuc_repeats'],
                   self.spec['gc_clamp_3p'][0], self.spec['gc_clamp_3p'][1],
                   self.spec['hairpin_tm_max'], self.spec['homodimer_tm_max'],
                   self.spec['homodimer_dg_min']))
