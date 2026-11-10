#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''Pre-NUPACK filter of primer candidates with the Olivia encoder.

Runs between get_primer() and optimize(). Each candidate is scored by the four
encoder heads (A, C_marg, C_inter, P); each score is mapped through its
empirical CDF over the candidate pool and the max is taken. The lowest-scoring
encoder_keep_frac of candidates are kept, per plex and direction.
'''


import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from math import ceil

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger('main')


# The trained checkpoint, in <repo>/results/encoder unless $OLIVIA_MODEL says
# otherwise. The repository root is two levels up from src/olivia/.
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
DEFAULT_MODEL_PATH = os.environ.get(
    'OLIVIA_MODEL', os.path.join(REPO_ROOT, 'results', 'encoder', 'encoder.pt'))

# The heads to fuse, in fusion_f1.csv's naming.
FUSION_HEADS = ('A', 'C_marg', 'C_inter', 'P')


class EncoderScorer(object):
    '''Wraps one loaded encoder checkpoint.'''
    def __init__(self, model_path=None, threads=1):
        import torch

        self.model_path = model_path or DEFAULT_MODEL_PATH
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                'encoder checkpoint not found at "%s". Pass encoder_model=<path> to tiling(), '
                'or set encoder_keep_frac=1.0 to skip the encoder filter.' % self.model_path)

        from olivia_encoder import density, encoder, properties, sequence

        self.torch = torch
        self.density = density
        self.encoder = encoder
        self.properties = properties
        self._to_tensor = sequence.sequence_to_tensor
        self._clean = sequence.clean_sequence

        torch.set_num_threads(max(1, int(threads)))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ckpt = torch.load(self.model_path, map_location='cpu', weights_only=False)
        self.seq_len = int(ckpt['seq_len'])
        self.limits = list(ckpt['limits'])
        self.widths = list(ckpt['dg_bin_widths'])
        self.bin_edges = [np.asarray(e, dtype=np.float32) for e in ckpt['bin_edges']]

        # Install the checkpoint's property list and Tm settings before rebuilding the
        # binners.
        properties.install_dg_properties(self.limits)
        if 'tm_conditions' in ckpt:
            properties.TM_CONDITIONS.update(ckpt['tm_conditions'])
        if 'tm_degenerate' in ckpt:
            properties.TM_DEGENERATE = ckpt['tm_degenerate']

        self.binners = properties.binners_from_checkpoint(ckpt)
        self.marg_scaler, self.inter_scaler = encoder.scalers_from_checkpoint(ckpt)

        model = encoder.build_model_from_checkpoint(ckpt)
        model.load_state_dict(ckpt['model_state_dict'])
        self.model = model.to(self.device).eval()

    def _bundle(self, seqs):
        '''Minimal stand-in for data.SplitBundle: a frame and the one-hot tensor.'''
        class _bundle(object):
            pass
        b = _bundle()
        b.frame = pd.DataFrame({'cleaned_primer': seqs})
        b.sequence = self.torch.tensor(np.stack(
            [self._to_tensor(s, self.seq_len) for s in seqs]))
        return b

    def embed(self, seqs, batch_size=256):
        """(N, 2*channels + pair_dim) sequence-level embeddings."""
        from torch.utils.data import DataLoader, TensorDataset

        seqs = [self._clean(s) for s in seqs]
        bundle = self._bundle(seqs)
        loader = DataLoader(TensorDataset(bundle.sequence), batch_size=batch_size, shuffle=False)
        out = []
        with self.torch.no_grad():
            for (seq,) in loader:
                seq = seq.to(self.device).float()
                out.append(self.model(seq, apply_mask=False)['pooled'].cpu().numpy())
        return np.concatenate(out, axis=0)

    def score(self, seqs, batch_size=256):
        '''Per-head scores for a list of primers, plus 'fused' (max of the per-head CDFs);
        higher is worse.
        '''
        from torch.utils.data import DataLoader, TensorDataset
        enc, props = self.encoder, self.properties

        seqs = [self._clean(s) for s in seqs]
        bundle = self._bundle(seqs)
        loader = DataLoader(TensorDataset(bundle.sequence), batch_size=batch_size, shuffle=False)

        a = enc.head_a_score_dg(self.model, loader, self.device,
                                self.bin_edges, self.limits, self.widths)
        # Six sequence classes; the two structural classes come from Head A's prediction
        # (no NUPACK).
        seq_cls = props.to_class_matrix(*props.sequence_property_values(bundle),
                                        self.binners[:props.SEQ_PROPERTY_COUNT])
        cls = enc.property_class_matrix(
            self.model, bundle.sequence, seq_cls, self.device,
            enc.dg_center_tensors(self.bin_edges, self.widths, self.device),
            enc.dg_edge_tensors(self.binners, self.device),
            batch_size=batch_size)
        cs = self.density.head_c_scores(
            self.density.head_c_terms(self.model.density, cls, self.device),
            self.marg_scaler, self.inter_scaler)
        p = enc.head_p_score(self.model, loader, self.device)['score']

        out = {'A': np.asarray(a['score'], dtype=np.float64),
               'C_marg': np.asarray(cs['c_marg'], dtype=np.float64),
               'C_inter': np.asarray(cs['c_inter'], dtype=np.float64),
               'P': np.asarray(p, dtype=np.float64)}
        out['fused'] = np.max(np.stack([_ecdf(out[h]) for h in FUSION_HEADS]), axis=0)
        return out


def _ecdf(x):
    '''Empirical CDF of x evaluated on x itself: the fraction of the pool scoring at or
    below each element.
    '''
    x = np.asarray(x, dtype=np.float64)
    ref = np.sort(x)
    return np.searchsorted(ref, x, side='right') / max(len(ref), 1)


def _row_sequences(seq_entry):
    '''One candidate row's sequences.'''
    if isinstance(seq_entry, (list, tuple)):
        return [str(s) for s in seq_entry]
    return [str(seq_entry)]


def filter_primer_candidates(all_plex_info, config):
    '''Keep the encoder_keep_frac least abnormal primer candidates per plex and direction.
    '''
    keep_frac = float(config.get('encoder_keep_frac', 0.5))
    if keep_frac >= 1.0:
        logger.info('encoder filter disabled (encoder_keep_frac=%.2f), keeping all primer candidates' % keep_frac)
        return all_plex_info
    if not 0.0 < keep_frac < 1.0:
        raise ValueError('encoder_keep_frac must be in (0, 1], got %r' % keep_frac)

    from time import time
    logger.info('Filtering primer candidates with the encoder (keeping best %.0f%%)...' % (100*keep_frac))
    tik = time()

    scorer = EncoderScorer(config.get('encoder_model'), threads=config.get('threads', 1))
    logger.info('encoder checkpoint: %s (device %s)' % (scorer.model_path, scorer.device))

    # One forward pass over every distinct sequence in the whole design.
    keys = [(plex_id, side) for plex_id in all_plex_info for side in ('fP', 'rP')]
    all_seq = set()
    for plex_id, side in keys:
        for seq_entry in all_plex_info[plex_id]['%s_candidate' % side]['seq']:
            all_seq.update(_row_sequences(seq_entry))
    all_seq = sorted(all_seq)
    if not all_seq:
        logger.info('no primer candidates to filter')
        return all_plex_info

    fused = scorer.score(all_seq)['fused']
    risk = dict(zip(all_seq, fused))

    n_before = 0
    n_after = 0
    for plex_id, side in keys:
        cand = all_plex_info[plex_id]['%s_candidate' % side]
        n_before += len(cand)
        # A degenerate candidate is only as usable as its worst expansion, so
        # the row takes the max over the concrete sequences it stands for.
        score = np.array([max(risk[s] for s in _row_sequences(e)) for e in cand['seq']])
        cand = cand.assign(encoder_score=score)
        # ceil, and never below 1: a plex with a single candidate keeps it, and
        # every plex still reaches optimize() with at least one fP-rP pair.
        k = max(1, ceil(len(cand) * keep_frac))
        # Ties broken by the original order (stable sort), which is descending
        # 3' position -- the same order get_primer() produced.
        kept = cand.sort_values('encoder_score', kind='stable').head(k).sort_index()
        all_plex_info[plex_id]['%s_candidate' % side] = kept
        n_after += len(kept)

    logger.info('primer candidates: %d -> %d (%.1f%% kept)'
                % (n_before, n_after, 100*n_after/max(n_before, 1)))
    logger.info('Finished in %.3fs' % (time()-tik))
    return all_plex_info
