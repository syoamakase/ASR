import copy
import itertools
import os
from struct import unpack, pack
from operator import itemgetter, attrgetter
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import hparams as hp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # encoder
        if hp.frame_stacking:
            input_size = hp.lmfb_dim * hp.frame_stacking
        else:
            input_size = hp.lmfb_dim
        self.bi_lstm = nn.LSTM(input_size=input_size, hidden_size=hp.num_hidden_nodes, num_layers=hp.num_encoder_layer, batch_first=True, dropout=0.2, bidirectional=True)
        # attention
        self.L_se = nn.Linear(hp.num_hidden_nodes, hp.num_hidden_nodes * 2, bias=False)
        self.L_he = nn.Linear(hp.num_hidden_nodes * 2, hp.num_hidden_nodes * 2)
        self.L_ee = nn.Linear(hp.num_hidden_nodes * 2, 1, bias=False)
        # conv attention
        self.F_conv1d = nn.Conv1d(1, 10, 100, stride=1, padding=50, bias=False)
        self.L_fe = nn.Linear(10, hp.num_hidden_nodes * 2, bias=False)

        # decoder
        self.L_sy = nn.Linear(hp.num_hidden_nodes, hp.num_hidden_nodes, bias=False)
        self.L_gy = nn.Linear(hp.num_hidden_nodes * 2, hp.num_hidden_nodes)
        self.L_yy = nn.Linear(hp.num_hidden_nodes, hp.num_classes)

        self.L_ys = nn.Linear(hp.num_classes, hp.num_hidden_nodes * 4 , bias=False)
        self.L_ss = nn.Linear(hp.num_hidden_nodes, hp.num_hidden_nodes * 4, bias=False)
        self.L_gs = nn.Linear(hp.num_hidden_nodes * 2, hp.num_hidden_nodes * 4)

    def forward(self, x, lengths, target):
        total_length = x.size(1)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        
        #self.bi_lstm.flatten_parameters()
        h, _ = self.bi_lstm(x)

        hbatch, lengths = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=total_length)
        # hbatch : hp.batch_size x num_frames x hp.num_hidden_nodes
        # lengths : hp.batch_size lengths array
        # target:  hp.batch_size x **num_labels** x hp.num_classes
        batch_size = hbatch.size(0)
        num_frames = hbatch.size(1)
        num_labels = target.size(1)

        e_mask = torch.ones((batch_size, num_frames, 1), device=DEVICE, requires_grad=False)

        s = torch.zeros((batch_size, hp.num_hidden_nodes), device=DEVICE, requires_grad=False)
        c = torch.zeros((batch_size, hp.num_hidden_nodes), device=DEVICE, requires_grad=False)

        youtput = torch.zeros((batch_size, num_labels, hp.num_classes), device=DEVICE, requires_grad=False)
        alpha = torch.zeros((batch_size, 1,  num_frames), device=DEVICE, requires_grad=False)

        for i, tmp in enumerate(lengths):
            if tmp < num_frames:
                e_mask.data[i, tmp:] = 0.0

        for step in range(num_labels):
            # (B, 1, width)
            tmpconv = self.F_conv1d(alpha)
            # (B, 10, channel)
            tmpconv = tmpconv.transpose(1, 2)[:, :num_frames, :]
            #
            tmpconv = self.L_fe(tmpconv)
            # BxTx2H
            e = torch.tanh(self.L_se(s).unsqueeze(1) + self.L_he(hbatch) + tmpconv)
            # BxT
            e = self.L_ee(e)
            # e_nonlin : hp.batch_size x num_frames
            e_nonlin = (e - e.max(1)[0].unsqueeze(1)).exp()
            # e_nonlin : hp.batch_size x num_frames
            e_nonlin = e_nonlin * e_mask

            alpha = e_nonlin / e_nonlin.sum(dim=1, keepdim=True)
            g = (alpha * hbatch).sum(dim=1)
            alpha = alpha.transpose(1, 2)

            # generate
            y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s)))

            # recurrency calcuate
            rec_input = self.L_ys(target[:,step,:]) + self.L_ss(s) + self.L_gs(g)
            s, c = self._func_lstm(rec_input, c)

            youtput[:,step] = y

        return youtput
    
    def decode(self, x, lengths):
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        h, _ = self.bi_lstm(x)

        hbatch, lengths = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        batch_size = 1
        num_frames = hbatch.size(1)
        e_mask = torch.ones((batch_size, num_frames), device=DEVICE, requires_grad=False)

        token_beam_sel = [([], [], 0.0, (torch.zeros((batch_size, hp.num_hidden_nodes), device=DEVICE, requires_grad=False),
                        torch.zeros((batch_size, hp.num_hidden_nodes), device=DEVICE, requires_grad=False),
                        torch.zeros((batch_size, num_frames), device=DEVICE, requires_grad=False)))]

        for i, tmp in enumerate(lengths):
            if tmp < num_frames:
                e_mask[i, tmp:] = 0.0

        for _ in range(hp.max_decoder_seq_len):
            token_beam_all = []

            for current_token in token_beam_sel:
                cand_seq, cand_bottle, cand_seq_score, (c, s, alpha) = current_token
                
                if len(cand_bottle) != 0:
                    tmp_bottle = copy.deepcopy(cand_bottle)
                else:
                    tmp_bottle = cand_bottle

                tmpconv = self.F_conv1d(alpha.unsqueeze(1))
                tmpconv = tmpconv.transpose(1, 2)[:, :num_frames, :]
                tmpconv = self.L_fe(tmpconv)
                # # TxBx2H
                e = torch.tanh(self.L_se(s).unsqueeze(1) + self.L_he(hbatch) + tmpconv)
                # # TxB
                e = self.L_ee(e).squeeze(2)

                e_nonlin = e.exp()
                e_nonlin = e_nonlin * e_mask

                alpha = (e_nonlin / e_nonlin.sum(dim=1, keepdim=True))

                g = (alpha.unsqueeze(2) * hbatch).sum(dim=1)
                # generate
                y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s)))
                bottle_feats = torch.tanh(self.L_gy(g) + self.L_sy(s)).detach().cpu().numpy()
                tmp_bottle.append(bottle_feats)

                if hp.score_func == 'log_softmax':
                    y = F.log_softmax(y, dim=1)
                elif hp.score_func == 'softmax':
                    y = F.softmax(y, dim=1)
                
                tmpy = y.clone()
                for _ in range(hp.beam_width):
                    bestidx = tmpy.data.argmax(1)
                    bestidx = bestidx.item()

                    tmpseq = cand_seq.copy()
                    tmpseq.append(bestidx)

                    tmpscore = cand_seq_score + tmpy.data[0][bestidx]
                    tmpy.data[0][bestidx] = -10000000000.0
                    target_for_t_estimated = torch.zeros((1, hp.num_classes), device=DEVICE, requires_grad=False)

                    target_for_t_estimated.data[0][bestidx] = 1.0

                    rec_input = self.L_ys(target_for_t_estimated) + self.L_ss(s) + self.L_gs(g)
                    tmps, tmpc = self._func_lstm(rec_input, c)

                    token_beam_all.append((tmpseq, tmp_bottle, tmpscore, (tmpc, tmps, alpha)))
            sorted_token_beam_all = sorted(token_beam_all, key=itemgetter(2), reverse=True)
            token_beam_sel = sorted_token_beam_all[:hp.beam_width]
            results = []
            if token_beam_sel[0][0][-1] == hp.eos_id:
                for character in token_beam_sel[0][0]:
                    results.append(character)
                break
        return results
    
    def _func_lstm(self, x, c):
        ingate, forgetgate, cellgate, outgate = x.chunk(4, 1)
        half = 0.5
        ingate = torch.tanh(ingate * half) * half + half
        forgetgate = torch.tanh(forgetgate * half) * half + half
        cellgate = torch.tanh(cellgate)
        outgate = torch.tanh(outgate * half) * half + half
        c_next = (forgetgate * c) + (ingate * cellgate)
        h = outgate * torch.tanh(c_next)
        return h, c