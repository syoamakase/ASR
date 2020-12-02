# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from Models.encoder import Encoder, WaveEncoder
from utils import hparams as hp


class CTCModel(nn.Module):
    def __init__(self, hp):
        super(CTCModel, self).__init__()
        self.hp = hp
        if self.hp.encoder_type == 'Wave':
            self.encoder = WaveEncoder()
        else:
            self.encoder = Encoder()
        
        self.decoder = nn.Linear(self.hp.num_hidden_nodes * 2, self.hp.num_classes)

    def forward(self, x, lengths, targets):
        hbatch = self.encoder(x, lengths)
        youtput = self.decoder(hbatch)

        return youtput
    
    def decode(self, x, lengths, model_lm=None, src_pos=None):
        with torch.no_grad():
            hbatch = self.encoder(x, lengths)
            decoder_output = self.decoder(hbatch)
            batch_size = hbatch.shape[0]

            results = []
            prev_id = hp.num_classes + 1
            for b in range(batch_size):
                results_batch = []
                for x in decoder_output[b].argmax(dim=1):
                    if int(x) != prev_id and int(x) != 0:
                        results_batch.append(int(x))
                    prev_id = int(x)
                results.append(results_batch)

        return results
            
    def align(self, x, lengths, targets=None, blank_id=0, space_id=1, sos_id=2, eos_id=1):
        with torch.no_grad():
            hbatch = self.encoder(x, lengths)
            # (B, T, V)
            decoder_output = self.decoder(hbatch)

        return decoder_output


class CTCForcedAligner(object):
    def __init__(self, blank=0):
        self.blank = blank
        self.log0 = -1e10
        self.log1 = 0

    def _computes_transition(self, prev_log_prob, path, path_lens, cum_log_prob, y, skip_accum=False):
        bs, max_path_len = path.size()
        mat = prev_log_prob.new_zeros(3, bs, max_path_len).fill_(self.log0)
        mat[0, :, :] = prev_log_prob
        mat[1, :, 1:] = prev_log_prob[:, :-1]
        mat[2, :, 2:] = prev_log_prob[:, :-2]
        # disable transition between the same symbols
        # (including blank-to-blank)
        same_transition = (path[:, :-2] == path[:, 2:])
        mat[2, :, 2:][same_transition] = self.log0
        log_prob = torch.logsumexp(mat, dim=0)
        outside = torch.arange(max_path_len, dtype=torch.int64) >= path_lens.unsqueeze(1)
        log_prob[outside] = self.log0
        if not skip_accum:
            cum_log_prob += log_prob
        batch_index = torch.arange(bs, dtype=torch.int64).unsqueeze(1)
        log_prob += y[batch_index, path]
        return log_prob

    def align(self, logits, elens, ys, ylens):

        bs, xmax, vocab = logits.size()

        # zero padding
        device_id = torch.cuda.device_of(logits).idx
        mask = make_pad_mask(elens, device_id)
        mask = mask.unsqueeze(2).repeat([1, 1, vocab])
        logits = logits.masked_fill_(mask == 0, self.log0)
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)  # `[T, B, vocab]`

        path = _label_to_path(ys, self.blank)
        path_lens = 2 * ylens.long() + 1

        ymax = ys.size(1)
        max_path_len = path.size(1)
        assert ys.size() == (bs, ymax), ys.size()
        assert path.size() == (bs, ymax * 2 + 1)

        alpha = log_probs.new_zeros(bs, max_path_len).fill_(self.log0)
        alpha[:, 0] = self.log1
        beta = alpha.clone()
        gamma = alpha.clone()
        # import pdb; pdb.set_trace()

        batch_index = torch.arange(bs, dtype=torch.int64).unsqueeze(1)
        seq_index = torch.arange(xmax, dtype=torch.int64).unsqueeze(1).unsqueeze(2)
        log_probs_fwd_bwd = log_probs[seq_index, batch_index, path]

        # forward algorithm
        for t in range(xmax):
            alpha = self._computes_transition(alpha, path, path_lens, log_probs_fwd_bwd[t], log_probs[t])

        # backward algorithm
        r_path = _flip_path(path, path_lens)
        log_probs_inv = _flip_label_probability(log_probs, elens.long())  # `[T, B, vocab]`
        log_probs_fwd_bwd = _flip_path_probability(log_probs_fwd_bwd, elens.long(), path_lens)  # `[T, B, 2*L+1]`
        for t in range(xmax):
            beta = self._computes_transition(beta, r_path, path_lens, log_probs_fwd_bwd[t], log_probs_inv[t])

        # pick up the best CTC path
        best_lattices = log_probs.new_zeros((bs, xmax), dtype=torch.int64)

        # forward algorithm
        log_probs_fwd_bwd = _flip_path_probability(log_probs_fwd_bwd, elens.long(), path_lens)
        for t in range(xmax):
            gamma = self._computes_transition(gamma, path, path_lens, log_probs_fwd_bwd[t], log_probs[t],
                                              skip_accum=True)

            # select paths where gamma is valid
            log_probs_fwd_bwd[t] = log_probs_fwd_bwd[t].masked_fill_(gamma == self.log0, self.log0)

            # pick up the best lattice
            offsets = log_probs_fwd_bwd[t].argmax(1)
            for b in range(bs):
                if t <= elens[b] - 1:
                    token_idx = path[b, offsets[b]]
                    best_lattices[b, t] = token_idx

            # remove the rest of paths
            gamma = log_probs.new_zeros(bs, max_path_len).fill_(self.log0)
            for b in range(bs):
                gamma[b, offsets[b]] = self.log1

        # pick up trigger points
        trigger_lattices = torch.zeros((bs, xmax), dtype=torch.int64)
        trigger_points = log_probs.new_zeros((bs, ymax + 1), dtype=torch.int32)  # +1 for <eos>
        for b in range(bs):
            n_triggers = 0
            trigger_points[b, ylens[b]] = elens[b] - 1  # for <eos>
            for t in range(elens[b]):
                token_idx = best_lattices[b, t]
                if token_idx == self.blank:
                    continue
                if not (t == 0 or token_idx != best_lattices[b, t - 1]):
                    continue

                # NOTE: select the most left trigger points
                trigger_lattices[b, t] = token_idx
                trigger_points[b, n_triggers] = t
                n_triggers += 1

        assert ylens.sum() == (trigger_lattices != 0).sum()
        return trigger_points

def _flip_path_probability(cum_log_prob, xlens, path_lens):
    """Flips a path probability matrix.
    This function returns a path probability matrix and flips it.
    ``cum_log_prob[i, b, t]`` stores log probability at ``i``-th input and
    at time ``t`` in a output sequence in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[i, j, k] = cum_log_prob[i + xlens[j], j, k + path_lens[j]]``
    Args:
        cum_log_prob (FloatTensor): `[T, B, 2*L+1]`
        xlens (LongTensor): `[B]`
        path_lens (LongTensor): `[B]`
    Returns:
        FloatTensor: `[T, B, 2*L+1]`
    """
    xmax, bs, max_path_len = cum_log_prob.size()
    rotate_input = ((torch.arange(xmax, dtype=torch.int64)[:, None] + xlens) % xmax)
    rotate_label = ((torch.arange(max_path_len, dtype=torch.int64) + path_lens[:, None]) % max_path_len)
    return torch.flip(cum_log_prob[rotate_input[:, :, None],
                                   torch.arange(bs, dtype=torch.int64)[None, :, None],
                                   rotate_label], dims=[0, 2])


def _flip_path(path, path_lens):
    """Flips label sequence.
    This function rotates a label sequence and flips it.
    ``path[b, t]`` stores a label at time ``t`` in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[b, t] = path[b, t + path_lens[b]]``
    .. ::
       a b c d .     . a b c d    d c b a .
       e f . . .  -> . . . e f -> f e . . .
       g h i j k     g h i j k    k j i h g
    Args:
        path (FloatTensor): `[B, 2*L+1]`
        path_lens (LongTensor): `[B]`
    Returns:
        FloatTensor: `[B, 2*L+1]`
    """
    bs = path.size(0)
    max_path_len = path.size(1)
    rotate = (torch.arange(max_path_len) + path_lens[:, None]) % max_path_len
    return torch.flip(path[torch.arange(bs, dtype=torch.int64)[:, None], rotate], dims=[1])


def _flip_label_probability(log_probs, xlens):
    """Flips a label probability matrix.
    This function rotates a label probability matrix and flips it.
    ``log_probs[i, b, l]`` stores log probability of label ``l`` at ``i``-th
    input in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[i, b, l] = log_probs[i + xlens[b], b, l]``
    Args:
        cum_log_prob (FloatTensor): `[T, B, vocab]`
        xlens (LongTensor): `[B]`
    Returns:
        FloatTensor: `[T, B, vocab]`
    """
    xmax, bs, vocab = log_probs.size()
    rotate = (torch.arange(xmax, dtype=torch.int64)[:, None] + xlens) % xmax
    return torch.flip(log_probs[rotate[:, :, None],
                                torch.arange(bs, dtype=torch.int64)[None, :, None],
                                torch.arange(vocab, dtype=torch.int64)[None, None, :]], dims=[0])

def make_pad_mask(seq_lens, device_id=-1):
    """Make mask for padding.
    Args:
        seq_lens (LongTensor): `[B]`
        device_id (int):
    Returns:
        mask (LongTensor): `[B, T]`
    """
    bs = seq_lens.size(0)
    max_time = max(seq_lens)

    seq_range = torch.arange(0, max_time, dtype=torch.long)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_time)
    seq_length_expand = seq_range_expand.new(seq_lens).unsqueeze(-1)
    mask = seq_range_expand < seq_length_expand

    if device_id >= 0:
        mask = mask.cuda(device_id)
    return mask

def _label_to_path(labels, blank):
    path = labels.new_zeros(labels.size(0), labels.size(1) * 2 + 1).fill_(blank).long()
    path[:, 1::2] = labels
    return path
