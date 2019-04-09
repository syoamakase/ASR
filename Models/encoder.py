# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import hparams as hp

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        if hp.frame_stacking:
            input_size = hp.lmfb_dim * hp.frame_stacking
        else:
            input_size = hp.lmfb_dim
        self.bi_lstm = nn.LSTM(input_size=input_size, hidden_size=hp.num_hidden_nodes, num_layers=hp.num_encoder_layer, \
                batch_first=True, dropout=hp.encoder_dropout, bidirectional=True)

    def forward(self, x, lengths):
        total_length = x.size(1)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        h, _ = self.bi_lstm(x)
        hbatch, lengths = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=total_length)
        return hbatch
    
class CNN_Encoder(nn.Module):
    def __init__(self):
        super(CNN_Encoder, self).__init__()
        