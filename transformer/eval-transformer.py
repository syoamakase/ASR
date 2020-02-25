# -*- coding: utf-8 -*-
"""
16/10/2018
Implementation of attention model using torch==1.0.0a0+8601b33

decoding
"""

import copy
import itertools
import os
import pdb
from struct import unpack, pack
import sys

import numpy as np
import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import Beam

FRAME_STACKING = 3#True
NUM_HIDDEN_NODES = 320
NUM_LAYERS = int(sys.argv[1]) # the number of bilstm layer
BATCH_SIZE = int(sys.argv[2]) # 30
NUM_CLASSES = int(sys.argv[3]) # 3260 / 34331
ATTENTION_OPTION = sys.argv[4] # "tanh" or "conv"
script_file = sys.argv[5]
model_name = sys.argv[6] # save dir name

model_path, network_epoch = model_name.split('/')
#epoch = int(network_epoch.replace('network.epoch',''))

#torch.manual_seed(7)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#print('NUM_LAYERS =', NUM_LAYERS)
#print('BATCH_SIZE =', BATCH_SIZE)
#print('NUM_CLASSES =', NUM_CLASSES)
#print('ATTENTION_OPTION =', ATTENTION_OPTION)
#print('script_file =', script_file)
#print('model_name =', model_name)

def load_dat(filename):
    """
    To read binary data in htk file.
    The htk file includes log mel-scale filter bank.

    Args:
        filename : file name to read htk file

    Returns:
        dat : 120 (means log mel-scale filter bank) x T (time frame)

    """
    fh = open(filename, "rb")
    spam = fh.read(12)
    nSamples, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)
    veclen = int(sampSize / 4)
    fh.seek(12, 0)
    dat = np.fromfile(fh, dtype=np.float32)
    dat = dat.reshape(int(len(dat) / veclen), veclen)
    dat = dat.byteswap()
    fh.close()
    return dat

def onehot(labels, num_output):
    """
    To make onehot vector.
    ex) labels : 3 -> [0, 0, 1, 0, ...]

    Args:
        labels : true label ID
        num_output : the number of entry

    Returns:
        utt_label : one hot vector.
    """
    utt_label = np.zeros((len(labels), num_output), dtype='float32')
    for i in range(len(labels)):
        utt_label[i][labels[i]] = 1.0
    return utt_label


def load_model(model_file):
    model_state = torch.load(model_file)
    is_multi_loading = True if torch.cuda.device_count() > 1 else False
    # This line may include bugs!!
    is_multi_loaded = True if 'module' in list(model_state.keys())[0] else False

    if is_multi_loaded is is_multi_loading:
        return model_state

    elif is_multi_loaded is False and is_multi_loading is True:
        new_model_state = {}
        for key in model_state.keys():
            new_model_state['module.'+key] = model_state[key]

        return new_model_state

    elif is_multi_loaded is True and is_multi_loading is False:
        new_model_state = {}
        for key in model_state.keys():
            new_model_state[key[7:]] = model_state[key]

        return new_model_state

    else:
        print('ERROR in load model')
        sys.exit(1)

def argsort(seq):
    return np.argsort(np.array(seq))[::-1].tolist()

# sorting lengths order
def sort_pad_decoding(xs, lengths):
    """
    To sort "lengths" order.
    This funtion is needed for "torch.nn.utils.rnn.pack_padded_sequence()"

    Args:
        xs : input feature. (BATCH SIZE, time frames, log mel-scale filter bank)
        lengths : the lengths of the input feature (BATCH SIZE)

    Returns:
        xs_tensor : "torch FloatTensor" of sorted xs
        lengths : sorted lenghts
    """
    arg_lengths = argsort(lengths)
    maxlen = max(lengths)
    xs_tensor = torch.zeros((BATCH_SIZE, maxlen, 120), dtype=torch.float32, requires_grad=True).to(DEVICE)
    lengths_tensor = torch.zeros((BATCH_SIZE), dtype=torch.int64).to(DEVICE)

    for i, i_sort in enumerate(arg_lengths):
        xs_tensor.data[i, 0:lengths[i_sort]] = torch.from_numpy(xs[i_sort])
        lengths_tensor.data[i] = lengths[i_sort]

    return xs_tensor, lengths_tensor


def init_weight(m):
    """
    To initialize weights and biases.
    """
    classname = m.__class__.__name__warmup_factor
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(-0.1, 0.1)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)

    if classname.find('LSTM') != -1:
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            if 'bias' in name:
                param.data.fill_(0)

    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None, ):

    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
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
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

##
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

# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

class Embedder(nn.Module):
    # copy
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    # copy
    def __init__(self, d_model, max_seq_len = 1500, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
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
        x = x + pe
        return self.dropout(x)


def get_clones(module, N):
    # copy
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    # copy
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = nn.Linear(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    # copy
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    # copy
    def __init__(self, src_vocab, trg_vocab, d_model, N_e, N_d, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N_e, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N_d, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        #print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output


def nopeak_mask(size):
    """
    npeak_mask(4)
    >> tensor([[[ 1,  0,  0,  0],
         [ 1,  1,  0,  0],
         [ 1,  1,  1,  0],
         [ 1,  1,  1,  1]]], dtype=torch.uint8)

    """
    np_mask = np.triu(np.ones((1, size, size)),
    k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0).to(DEVICE)
    return np_mask

def create_masks(src, trg, src_pad, trg_pad):
    
    src_mask = (src != src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != trg_pad).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size)
        if trg.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask
        
    else:
        trg_mask = None
    return src_mask, trg_mask

# initialize pytorch
    # def __init__(self, n_src_vocab, n_tgt_vocab, len_max_seq, d_word_vec=512, d_model=512, d_inner=2048, n_layers=6, n_head=8,
    #                 d_k=64, d_v=64, dropout=0.1, tgt_emb_prj_weight_sharing=True, emb_src_tgt_weight_sharing=True):
#model = Transformer(n_src_vocab=0, n_tgt_vocab=NUM_CLASSES+1, len_max_seq=1000)
# translator = Translator(model_name)

# multi-gpu setup
#if torch.cuda.device_count() > 1:
    # multi-gpu configuration
#    ngpu = torch.cuda.device_count()
#    device_ids = list(range(ngpu))
#    model = torch.nn.DataParallel(model, device_ids)
#    model.cuda()
#else:
#    model.to(DEVICE)
#
#model.eval()
#model.train()
model = Transformer(src_vocab=FRAME_STACKING*40, trg_vocab=NUM_CLASSES+1, N_e=12, N_d=6, heads=4, d_model=256, dropout=0.0)
#model = Transformer(src_vocab=120, trg_vocab=NUM_CLASSES+1, N=6, heads=16, d_model=1024, dropout=0.1)
model.to(DEVICE)
model.eval()

# load
model.load_state_dict(load_model(model_name))
#optimizer.load_state_dict(torch.load("model.word.mb40_LS-SS_fix/network.optimizer.epoch30"))

script_buf = []
with open(script_file) as f:
    for line in f:
        script_buf.append(line)

#fout = open(os.path.join(model_path,'e{}.txt'.format(epoch)), 'w')
num_mb = len(script_buf) // BATCH_SIZE
maxlen = 0
for i in range(num_mb):
    # input lmfb (B x T x 120)
    xs = []
    # target symbols
    ts = []
    # onehot vector of target symbols (B x L x NUM_CLASSES)
    ts_onehot = []
    # vector of target symbols for label smoothing (B x L x NUM_CLASSES)
    ts_onehot_LS = []
    # input lengths
    lengths = []
    ts_lengths = []
    for j in range(BATCH_SIZE):
        s = script_buf[i*BATCH_SIZE+j].strip()
        if len(s.split(' ')) == 1:
            x_file = s
        else:
            x_file, laborg = s.split(' ', 1)
        if '.htk' in x_file:
            cpudat = load_dat(x_file)
            cpudat = cpudat[:, :40]
        elif '.npy'in x_file:
            cpudat = np.load(x_file)

        #print("{}".format(x_file),end=' ')
        if FRAME_STACKING:
            newlen = int(cpudat.shape[0] / FRAME_STACKING)
            cpudat = cpudat[:FRAME_STACKING * newlen, :]
            cpudat = np.reshape(cpudat, (newlen, FRAME_STACKING, 40))
            cpudat = np.reshape(cpudat, (newlen, FRAME_STACKING * 40)).astype(np.float32)
        lengths.append(newlen)
        xs.append(cpudat)
    #xs , _= sort_pad_decoding(xs, lengths)
    xs_dummy = []
    src_pad = 0
    for i in range(len(xs)):
        xs_dummy.append([1] * lengths[i])
    # 29/10/2018 xs and xs_pos is confirmed
    src_seq = np.zeros((BATCH_SIZE, max(lengths), FRAME_STACKING * 40))
    for i in range(len(xs)):
        src_seq[i, :lengths[i], :] = xs[i]
    src_seq_dummy = np.array([inst + [src_pad] * (max(lengths) - len(inst)) for inst in xs_dummy])

    src_seq = torch.from_numpy(src_seq).to(DEVICE).float()
    src_seq_dummy = torch.from_numpy(src_seq_dummy).to(DEVICE).long()
    # 29/10/2018 xs and xs_pos is confirmed

    youtput_in_Variable = Beam.beam_search(src_seq, src_seq_dummy, model)
    print("{} {}".format(x_file.strip(), youtput_in_Variable))
    #if not youtput_in_Variable.strip() == '':
    #    fout.write("{} {}\n".format(x_file.strip(), youtput_in_Variable))

    sys.stdout.flush()
#fout.close()
