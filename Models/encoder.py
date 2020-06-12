# -*- coding: utf-8 -*-

import numpy as np
import scipy
import torch
import torch.nn.functional as F
import torch.nn as nn
import warnings

from utils import hparams as hp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Use warning filter because lstm warning displays per iter. 
warnings.simplefilter('ignore')

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        if hp.frame_stacking > 1:
            input_size = hp.lmfb_dim * hp.frame_stacking
        elif hp.encoder_type == 'CNN':
            self.cnn_encoder = CNN_embedding(hp.lmfb_dim, 128)
            input_size = hp.lmfb_dim//4 * 128
        else:
            input_size = hp.lmfb_dim * hp.frame_stacking
        self.bi_lstm = nn.LSTM(input_size=input_size, hidden_size=hp.num_hidden_nodes_encoder, num_layers=hp.num_encoder_layer, \
                batch_first=True, dropout=hp.encoder_dropout, bidirectional=True)

    def forward(self, x, lengths):
        if hp.encoder_type == 'CNN':
            x, lengths = self.cnn_encoder(x, lengths)
        total_length = x.shape[1]
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.bi_lstm(x)
        hbatch, lengths = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=total_length)
        return hbatch
    
class CNN_embedding(nn.Module):
    def __init__(self, idim, odim):
        super().__init__()

        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.batch_norm_1 = nn.BatchNorm2d(odim)
        # self.layer_norm_1 = nn.LayerNorm(odim * idim)
        
        self.conv2_1 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        # self.batch_norm_2 = nn.BatchNorm2d(odim)
        self.instance_norm = nn.InstanceNorm2d(128)
        # self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout = nn.Dropout(0.2)

        #self.out = nn.Linear(odim * idim, hp.num_hidden_nodes_encoder)

    def forward(self, x, x_length):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1_1(x))
        x = torch.relu(self.conv1_2(x))
        x = F.max_pool2d(x, 2, stride=2, ceil_mode=True)

        x = torch.relu(self.conv2_1(x))
        x = torch.relu(self.conv2_2(x))
        x = F.max_pool2d(x, 2, stride=2, ceil_mode=True)
        x = self.instance_norm(x)
        b, c, t, f = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c*f)
        # x = self.layer_norm_1(x.transpose(1, 2).contiguous().view(b, t, c*f)).view(b, t, c, f).transpose(1, 2)
        # x = self.conv_2(x)
        # x = self.batch_norm_2(x)
        # b, c, t, f = x.size()
        # x = self.layer_norm_2(x.transpose(1, 2).contiguous().view(b, t, c*f))
        # x = torch.relu(x)
        # x = self.dropout(x)
        # x = self.pool_1(x)
        # x = self.pool_2(x)
        #x = self.out(x)

        return x, torch.ceil(torch.ceil(x_length/2.0)/2.0)

class WaveEncoder(nn.Module):
    def __init__(self, hp):
        super(WaveEncoder, self).__init__()
        ## frond-end part
        self.epsilon = 1e-8
        # Like preemphasis filter
        self.preemp = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, bias=False)
        # init
        tmp = torch.zeros((1,1,2)).to(DEVICE)
        tmp.data[:,:,0] = -0.97
        tmp.data[:,:,1] = 1
        self.preemp.weight.data = torch.tensor(tmp)

        # if 16kHz
        self.comp = nn.Conv1d(in_channels=1, out_channels=80, kernel_size=400, stride=1, padding=0, bias=False)
        nn.init.kaiming_normal_(self.comp.weight.data)

        # B x 400 (0.01s = 10ms)
        tmp = np.zeros((40, 1, 400))
        tmp[:, :] = scipy.hanning(400 + 1)[:-1]
        tmp = tmp * tmp

        K = torch.tensor(tmp, dtype=torch.float).to(DEVICE)

        self.lowpass_weight = K

        self.instancenorm = nn.InstanceNorm1d(40)

        # encoder part
        if hp.frame_stacking:
            input_size = hp.lmfb_dim * hp.frame_stacking
        else:
            input_size = hp.lmfb_dim

        self.bi_lstm = nn.LSTM(input_size=input_size, hidden_size=hp.num_hidden_nodes, num_layers=hp.num_encoder_layer, \
                batch_first=True, dropout=hp.encoder_dropout, bidirectional=True)

    def forward(self, x_waveform, lengths_waveform):

        x_preemp = self.preemp(x_waveform.permute(0,2,1))
        x_comp = self.comp(x_preemp)

        x_even = x_comp[:, 0::2, :]
        x_odd = x_comp[:, 1::2, :]
        x_abs = torch.sqrt(x_even * x_even + x_odd * x_odd + self.epsilon)

        x_lowpass = F.conv1d(x_abs, self.lowpass_weight, stride=160, groups=40)

        x_log = torch.log(1.0 + torch.abs(x_lowpass))

        x_norm = self.instancenorm(x_log).permute(0,2,1)
        
        x_lengths = lengths_waveform - 1
        x_lengths = (x_lengths - (400 - 1)) // 1
        x_lengths = (x_lengths - (400 - 160)) // 160

        seqlen = x_norm.shape[1]
        
        if hp.frame_stacking:
            if seqlen % 3 == 0:
               x_norm = torch.cat((x_norm[:, 0::3], x_norm[:, 1::3], x_norm[:, 2::3, :]), dim=2)
            elif seqlen % 3 == 1:
                x_norm = torch.cat((x_norm[:, 0:-1:3,:], x_norm[:, 1::3, :], x_norm[:, 2::3, :]), dim=2)
            elif seqlen % 3 == 2:
                x_norm = torch.cat((x_norm[:, 0:-2:3,:], x_norm[:, 1:-1:3, :], x_norm[:, 2::3, :]), dim=2)
            
            x_lengths /= 3
        
        x = nn.utils.rnn.pack_padded_sequence(x_norm, x_lengths.tolist(), batch_first=True)

        h, (_, _) = self.bi_lstm(x)

        hbatch, lengths = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)

        return hbatch
