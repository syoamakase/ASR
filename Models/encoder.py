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
        # encoder_cnn
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = (3, 3), 
                        stride = (2, 2), padding = (1, 0)) 
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3, 3), 
                        stride = (2, 2), padding = (1, 0)) 
        self.conv2_bn = nn.BatchNorm2d(32)
        # encoder
        self.bi_lstm = nn.LSTM(input_size=640, hidden_size=hp.num_hidden_nodes, num_layers=hp.num_encoder_layer,
                         batch_first=True, dropout=hp.encoder_dropout, bidirectional=True)

    def forward(self, x, lengths):
        batch_size = x.size(0)
        conv_out = self.conv1(x.permute(0, 2, 3, 1)) 
        batched = self.conv1_bn(conv_out)
        activated = F.relu(batched)
        conv_out = self.conv2(activated)
        batched = self.conv2_bn(conv_out)
        activated = F.relu(batched)

        cnnout = activated.permute(0, 3, 1, 2).reshape(batch_size, activated.size(3), -1) 

        newlengths = []
        for xlen in lengths.cpu().numpy():
            q1, mod1 = divmod(xlen, 2)
            if mod1 == 0:
                xlen1 = xlen // 2 - 1
                q2, mod2 = divmod(xlen1, 2)
                if mod2 == 0:
                    xlen2 = xlen1 // 2 - 1
                else:
                    xlen2 = (xlen1 - 1) // 2
            else:
                xlen1 = (xlen - 1) // 2
                q2, mod2 = divmod(xlen1, 2)
                if mod2 == 0:
                    xlen2 = xlen1 // 2 - 1
                else:
                    xlen2 = (xlen1 - 1) // 2
            newlengths.append(xlen2)

        cnnout_packed = nn.utils.rnn.pack_padded_sequence(cnnout, newlengths, batch_first=True)

        h, _ = self.bi_lstm(cnnout_packed)

        hbatch, newlengths = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        return hbatch

