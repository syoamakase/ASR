# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class Attention(nn.Module):
    """
    Attention mechanism based on content-based model [Chorowski+, 2015]
    """
    def __init__(self, hp):
        super(Attention, self).__init__()
        self.num_decoder_hidden_nodes = hp.num_hidden_nodes_decoder
        if hp.encoder_type == 'Conformer':
            self.num_encoder_hidden_nodes = hp.num_hidden_nodes_encoder
        else:
            self.num_encoder_hidden_nodes = hp.num_hidden_nodes_encoder * 2 
        # attention
        self.L_se = nn.Linear(self.num_decoder_hidden_nodes, self.num_decoder_hidden_nodes * 2, bias=False)
        self.L_he = nn.Linear(self.num_encoder_hidden_nodes, self.num_decoder_hidden_nodes * 2)
        self.L_ee = nn.Linear(self.num_decoder_hidden_nodes * 2, 1, bias=False)
        self.L_fe = nn.Linear(10, self.num_decoder_hidden_nodes * 2, bias=False)
        self.F_conv1d = nn.Conv1d(1, 10, 100, stride=1, padding=50, bias=False)

    def forward(self, s, hbatch, alpha, e_mask):
        num_frames = hbatch.size(1)
        tmpconv = self.F_conv1d(alpha)
        tmpconv = tmpconv.transpose(1, 2)[:, :num_frames, :]
        tmpconv = self.L_fe(tmpconv)
        # BxTx2H
        e = torch.tanh(self.L_se(s).unsqueeze(1) + self.L_he(hbatch) + tmpconv)
        # BxT
        e = self.L_ee(e)
        e_nonlin = (e - e.max(1)[0].unsqueeze(1)).exp()
        e_nonlin = e_nonlin * e_mask

        alpha = e_nonlin / e_nonlin.sum(dim=1, keepdim=True)
        g = (alpha * hbatch).sum(dim=1)
        alpha = alpha.transpose(1, 2)

        return g, alpha
