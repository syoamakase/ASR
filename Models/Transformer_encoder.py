# -*- coding: utf-8 -*-
import argparse
import copy
import os
from struct import unpack, pack
import sys
import time

import random
import numpy as np
import math

import torch
import torch.nn.functional as F
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
       
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):

        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)       
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, pos_emb, mask=None):

        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        #q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        # relative pos
        n_batch_pos = pos_emb.shape[0]
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1,2)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1,2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1,2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2,-1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2,-1))
        matrix_bd = self.rel_shift(matrix_bd)

        matrix = matrix_ac+matrix_bd
        scores, attn = self.attention(matrix, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output, attn

    def rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def attention(self, matrix, v, d_k, mask=None, dropout=None):
    
        scores = matrix /  math.sqrt(d_k)
    
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = scores.masked_fill(mask == 0, -1e4)
            attn = torch.softmax(attn, dim=-1) # (batch, head, time1, time2)
        else:
            attn = F.softmax(attn, dim=-1)
        if dropout is not None:
            attn = dropout(attn)
    
        output = torch.matmul(attn, v)
        return output, attn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

        nn.init.xavier_uniform_(self.linear_1.weight,
            gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.linear_2.weight,
            gain=nn.init.calculate_gain('linear'))

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, groups=in_channels)
        self.conv_out = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x):
        x = F.pad(x, self.padding)
        x = self.conv(x)
        return self.conv_out(x)

class FeedForwardConformer(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.swish = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)

        return x

class ConvolutionModule(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        causal = False
        kernel_size = 31
        padding = self.calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_model*2, kernel_size=1)

        self.depth_conv1 = DepthwiseConv(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, padding=padding)
        ####
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (B x T x H) -> (B x H x T)
        x = self.layer_norm(x).transpose(1,2)
        x = self.pointwise_conv1(x)
        out, gate = x.chunk(2, dim=1)
        x = out * gate.sigmoid()
        x = self.depth_conv1(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x).transpose(1,2)
        x = self.dropout(x)

        return x

    def calc_same_padding(self, kernel_size):
        pad = kernel_size // 2
        return (pad, pad - (kernel_size + 1) % 2)

class ConformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.ff_1 = FeedForwardConformer(d_model, dropout=dropout)
        self.norm = nn.LayerNorm(d_model) #Norm(d_model)
        self.attn = RelativeMultiHeadAttention(heads, d_model, dropout=dropout)
        self.conv_module = ConvolutionModule(d_model, dropout=dropout)
        self.ff_2 = FeedForwardConformer(d_model, dropout=dropout)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, pe, mask):
        x = x + 0.5 * self.ff_1(x)
        res = x
        x = self.norm(x)
        x, attn_enc_enc = self.attn(x,x,x,pe,mask)
        x = res + self.dropout_1(x)
        x = x + self.conv_module(x)
        x = x + self.dropout_2(self.ff_2(x))
        return x, attn_enc_enc

class PositionalEncoder(nn.Module):
    # copy
    def __init__(self, d_model, max_seq_len=1500, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        self.xscale = 1 # math.sqrt(d_model)

        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        if hp.pe_alpha:
            x = x + self.alpha * pe
        else:
            x = x + pe
        return self.dropout(x)

class RelativePositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.xscale = 1 #math.sqrt(d_model)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
        
    def forward(self, x):
        x = x * self.xscale
        seq_len = x.shape[1]
        pe = self.pe[:,:seq_len].to(x.device)

        return self.dropout(x), self.dropout(pe)

class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def forward(self, *args):
        """Repeat."""
        for m in self:
            args = m(*args)
        return args


def repeat(N, fn):
    """Repeat module N times.
    :param int N: repeat time
    :param function fn: function to generate module
    :return: repeated modules
    :rtype: MultiSequential
    """
    return MultiSequential(*[fn() for _ in range(N)])

def get_clones(module, N):
    # copy
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class ConformerEncoder(nn.Module):
    # copy
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.heads = heads
        self.pe = RelativePositionalEncoder(d_model, dropout=dropout)
        self.layers = repeat(N, lambda: ConformerEncoderLayer(d_model, heads, dropout))
        #self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask):
        x, pe = self.pe(src)
        b, t, _ = x.shape
        attns_enc = torch.zeros((b, self.N, self.heads, t, t), device=x.device)
        for i in range(self.N):
            x, attn_enc = self.layers[i](x, pe, mask)
            attns_enc[:,i] = attn_enc.detach()
        return x, attns_enc

class Encoder(nn.Module):
    # copy
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        #self.embed = nn.Linear(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        #self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.layers = repeat(N, lambda: EncoderLayer(d_model, heads, dropout)) 
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask):
        #x = self.embed(src)
        x = self.pe(src)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class CNN_embedding(nn.Module):
    def __init__(self, idim, channels, odim):
        super().__init__()

        self.conv = torch.nn.Sequential(
            nn.Conv2d(1, channels, 3, 2),
            nn.ReLU(),
            torch.nn.Conv2d(channels, channels, 3, 2),
            nn.ReLU()
        )
        self.out = torch.nn.Sequential(
            nn.Linear(channels * (((idim - 1) // 2 - 1) // 2), odim, bias=False),
        )

    def forward(self, x, x_mask):
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c*f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]


class CNN_embedding_v2(nn.Module):
    def __init__(self, idim, odim):
        super().__init__()

        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv2_1 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.instance_norm = nn.InstanceNorm2d(odim)
        self.out = nn.Linear(128 * (idim // 2 // 2), odim, bias=False)

    def forward(self, x, x_mask):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1_1(x))
        x = torch.relu(self.conv1_2(x))
        x = F.max_pool2d(x, 2, stride=2, ceil_mode=True)

        x = torch.relu(self.conv2_1(x))
        x = torch.relu(self.conv2_2(x))
        x = F.max_pool2d(x, 2, stride=2, ceil_mode=True)
        x = self.instance_norm(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c*f))
        if x_mask is None:
            return x, None
        x_mask = x_mask[:,:,::2] if x_mask.shape[-1]%2 == 1 else x_mask[:,:,:-1:2]
        x_mask = x_mask[:,:,::2] if x_mask.shape[-1]%2 == 1 else x_mask[:,:,:-1:2]
        return x, x_mask #x_mask[:, :, :-2:2][:, :, :-2:2] #torch.ceil(torch.ceil(x_length/2.0)/2.0)


class TransformerEncoder(nn.Module):
    def __init__(self, src_vocab, d_model, N_e, heads, dropout):
        super().__init__()
        self.cnn_encoder = CNN_embedding(src_vocab, d_model, d_model)
        self.encoder = ConformerEncoder(d_model, N_e, heads, dropout)

    def forward(self, src, src_pos):
        src_mask = (src_pos != 0).unsqueeze(-2)
        src, src_mask = self.cnn_encoder(src, src_mask)
        e_outputs, attn_enc = self.encoder(src, src_mask)
        return e_outputs

class TransformerTextEncoder(nn.Module):
    def __init__(self, src_vocab, d_model, N_e, heads, dropout):
        super().__init__()
        self.encoder = Encoder(d_model, N_e, heads, dropout)

    def forward(self, src, src_pos):
        src_mask = (src_pos != 0).unsqueeze(-2)
        #src, src_mask = self.cnn_encoder(src, src_mask)
        e_outputs, attn_enc = self.encoder(src, src_mask)
        return e_outputs


def nopeak_mask(size):
    """
    npeak_mask(4)
    >> tensor([[[ 1,  0,  0,  0],
         [ 1,  1,  0,  0],
         [ 1,  1,  1,  0],
         [ 1,  1,  1,  1]]], dtype=torch.uint8)

    """
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = (torch.from_numpy(np_mask) == 0).to(DEVICE)
    return np_mask


def create_masks(src_pos, src_pad=0):
    src_mask = (src_pos != 0).unsqueeze(-2)

    if trg_pos is not None:
        trg_mask = (trg_pos != trg_pad).unsqueeze(-2)
        size = trg_pos.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size)
        if trg_pos.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None
    return src_mask, trg_mask


def get_learning_rate(step):
    d_model = 256
    return warmup_factor * min(step ** -0.5, step * warmup_step ** -1.5) * (d_model ** -0.5)


if __name__ == '__main__':
    model = TransformerEncoder(80, 256, 12, 4, 0.1)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('params = {0:.2f}M'.format(pytorch_total_params / 1000 / 1000))
