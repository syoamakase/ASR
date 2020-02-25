# -*- coding: utf-8 -*-
"""
16/10/2018
Implementation of transformer model using torch==1.0.0a0+8601b33

reference:
https://github.com/SamLynnEvans/Transformer.git
testtest

"""
import copy
import itertools
import os
from struct import unpack, pack
import sys

import random
import numpy as np
import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from tqdm import tqdm
#from Datasets_csj import dataset_transformer
from Datasets import dataset_transformer
from torch.utils.data import DataLoader
#from pytorch_memlab import profile, MemReporter
import utils_specaug

import hparams as hp

FRAME_STACKING = 1 # False
#NUM_CLASSES = int(sys.argv[1]) # 3260 / 34331
save_dir = sys.argv[1] # save dir name
label_smoothing = True #False
clip = 1.0

random.seed(77)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('max_seqlen =', hp.max_seqlen)
print('NUM_CLASSES =', hp.vocab_size)
print('label smoothing', label_smoothing)
print('save_dir =', save_dir)
print('PID =', os.getpid())
print('HOST =', os.uname()[1])
print('gradient clip =', clip)
print('Frame stacking', FRAME_STACKING)

def load_model(model_file):
    """
    To load the both of multi-gpu model and single gpu model.
    """
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

def init_weight(m):
    """
    To initialize weights and biases.
    """
    classname = m.__class__.__name__
    if classname.find('linear') != -1:
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

def attention(q, k, v, d_k, mask=None, dropout=None):

    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e10)
        #mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
        #min_value = float(np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
        #scores = scores.masked_fill(mask, min_value)
        scores = torch.softmax(scores, dim=-1) # (batch, head, time1, time2)
    else:
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

        nn.init.xavier_uniform_(self.q_linear.weight,
            gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.v_linear.weight,
            gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.k_linear.weight,
            gain=nn.init.calculate_gain('linear'))

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
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
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

class FeedForward_conv_hidden_relu(nn.Module):
    def __init__(self, d_model, d_ff=2048, kernel_size=3, second_kernel_size=31, dropout = 0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.f_1 = nn.Conv1d(d_model, d_ff, kernel_size=1) #kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.f_2 = nn.Conv1d(d_ff, d_model, kernel_size=1) #second_kernel_size)

    def forward(self, x):
        x = self.dropout(F.relu(self.f_1(x.transpose(1,2))))
        x = self.f_2(x).transpose(2, 1)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model) #Norm(d_model)
        self.norm_2 = nn.LayerNorm(d_model) #Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    #@profile
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
        self.norm_1 = nn.LayerNorm(d_model) #Norm(d_model)
        self.norm_2 = nn.LayerNorm(d_model) #Norm(d_model)
        self.norm_3 = nn.LayerNorm(d_model) #Norm(d_model)
        #self.norm_1 = Norm(d_model)
        #self.norm_2 = Norm(d_model)
        #self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    #@profile
    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, \
        src_mask))
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

class PositionalEncoder_enc(nn.Module):
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

class PositionalEncoder_dec(nn.Module):
    # copy
    def __init__(self, d_model, max_seq_len = 100, dropout = 0.1):
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

class PositionalEncoder(nn.Module):
    # copy
    def __init__(self, d_model, max_seq_len = 1500, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i

        self.xscale = math.sqrt(d_model)

        pe = torch.zeros(max_seq_len, d_model)
        #self.alpha = nn.Parameter(torch.ones(1))
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
        #x = x + self.alpha * pe
        x = x + pe
        return self.dropout(x)

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

    #@profile
    #def forward(self, src, mask):
    def forward(self, src, mask):
        #x = self.embed(src)
        x = self.pe(src)
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
        #self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.layers = repeat(N, lambda: DecoderLayer(d_model, heads, dropout))
        self.norm = nn.LayerNorm(d_model)

    #@profile
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class CNN_embedding(nn.Module):
    def __init__(self, idim, odim):
        super().__init__()

        self.conv = torch.nn.Sequential(
            nn.Conv2d(1, odim, 3,2),
            nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            nn.ReLU()
        )
        self.out = torch.nn.Sequential(
            nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),        
        )
        nn.init.xavier_uniform_(self.conv[0].weight,
            gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.conv[2].weight,
            gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.out[0].weight,
            gain=nn.init.calculate_gain('linear'))
        

    def forward(self, x, x_mask):
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c*f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

class Transformer(nn.Module):
    # copy
    def __init__(self, src_vocab, trg_vocab, d_model, N_e, N_d, heads, dropout):
        super().__init__()
        self.cnn_encoder = CNN_embedding(src_vocab, d_model)
        self.encoder = Encoder(src_vocab, d_model, N_e, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N_d, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    #@profile
    def forward(self, src, trg, src_mask, trg_mask):
        # import pdb;pdb.set_trace()
        src, src_mask = self.cnn_encoder(src, src_mask)
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        outputs = self.out(d_output)
        return outputs

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

def create_masks(src_pos, trg_pos, src_pad=0, trg_pad=0):
    src_mask = (src_pos != src_pad).unsqueeze(-2)

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

def spec_aug(x, x_lengths):
    # x is B x T x F
    aug_F = 15
    aug_T = 100
    #aug_mT = 2
    batch_size = len(x_lengths)
    for i in range(batch_size):
        x_frames = x_lengths[i]

        aug_f = np.random.randint(0, aug_F)
        aug_f0 = np.random.randint(0, 40 - aug_f)

        if x_frames > aug_T:
            duration = np.random.randint(0, aug_T)
        else:
            duration = np.random.randint(0, x_frames-1)
        start_t = np.random.randint(0, x_frames - duration)

        x[i, start_t:start_t+duration, :] = 0.0
        x[i, :, aug_f:aug_f+aug_f0] = 0.0

    return x

def get_learning_rate(step):
    d_model = 256
    return warmup_factor * min(step ** -0.5, step * warmup_step ** -1.5) * (d_model ** -0.5)

# initialize pytorch
model = Transformer(src_vocab=40*FRAME_STACKING, trg_vocab=hp.vocab_size+1, N_e=12, N_d=6, heads=4, d_model=256, dropout=0.1)
#model = Transformer(src_vocab=120, trg_vocab=NUM_CLASSES+1, N=6, heads=16, d_model=1024, dropout=0.3)
#model.apply(init_weight)
#print(model)

# initialize weights
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

# multi-gpu setup
if torch.cuda.device_count() > 1:
    # multi-gpu configuration
    ngpu = torch.cuda.device_count()
    device_ids = list(range(ngpu))
    model = torch.nn.DataParallel(model, device_ids)
    model.cuda()
else:
    model.to(DEVICE)

# model.eval()
model.train()

# The default parameters make the performance of the A2W model better
max_lr = 1e-3
warmup_step = hp.warmup_step # 4000
warmup_factor = hp.warmup_factor #10.0 # 1.0
optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, betas=(0.9, 0.98), eps=1e-9)

# T_max = 282237 // 40

os.makedirs(save_dir, exist_ok=True)
#print('Weight decay = ', optimizer.param_groups[0]['weight_decay'])

#reporter = MemReporter(model)
#reporter.report(verbose=True)

# load
#loaded_epoch = 40
#model.load_state_dict(load_model("transformer.tedlium_10kbpe.50k.warmup25000_factor5.0.cnn_downsample/network.epoch{}".format(loaded_epoch)))
#optimizer.load_state_dict(torch.load("transformer.tedlium_10kbpe.50k.warmup25000_factor5.0.cnn_downsample/network.optimizer.epoch{}".format(loaded_epoch)))
#step = 1540 * loaded_epoch #* (len(script_buf) // BATCH_SIZE")
step = 1

#datasets = dataset_transformer.get_dataset('/home/ueno/Datasets/tedlium/script.bpe10k.sort_xlen.r2')
datasets = dataset_transformer.get_dataset(hp.script_file)
collate_fn_transformer = dataset_transformer.collate_fn_transformer

assert (hp.batch_size is None) != (hp.max_seqlen is None)

for epoch in range(0, 100):
    src_pad = 0
    trg_pad = hp.vocab_size 

    if hp.batch_size is not None:
        sampler = dataset_transformer.NumBatchSampler(datasets, 120)
    elif hp.max_seqlen is not None:
        sampler = dataset_transformer.LengthsBatchSampler(datasets, hp.max_seqlen, hp.lengths_file)
    #sampler = dataset_transformer.LengthsBatchSampler(datasets, 105000, '/home/ueno/Datasets/csj/lengths.npy')
    #sampler = dataset_transformer.LengthsBatchSampler(datasets, hp.max_seqlen, '/home/ueno/Datasets/csj/lengths.npy')
    dataloader = DataLoader(datasets, batch_sampler=sampler, num_workers=4, collate_fn=collate_fn_transformer)

    #scheduler.step(epoch)
    #pbar = tqdm(dataloader)
    #for d in pbar:
    for d in dataloader: 
        lr = get_learning_rate(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        text, mel_input, pos_text, pos_mel, text_lengths = d

        #T = min(mel_input.shape[1] // 2 - 1, 40)
        #mel_input = utils_specaug.time_mask(utils_specaug.freq_mask(mel_input.clone().transpose(1, 2), num_masks=1), T=T, num_masks=1).transpose(1,2)
        #mel_input = spec_aug(mel_input.clone())
        text = text.to(DEVICE)
        mel_input = mel_input.to(DEVICE)
        pos_text = pos_text.to(DEVICE)
        pos_mel = pos_mel.to(DEVICE)
        text_lengths = text_lengths.to(DEVICE)

        batch_size = mel_input.shape[0]

        text_input = text[:, :-1]
        src_mask, trg_mask = create_masks(pos_mel, pos_text[:, :-1])

        youtputs = model(mel_input, text_input, src_mask, trg_mask)

        ys = text[:,1:].contiguous().view(-1)
        optimizer.zero_grad()

        loss = 0.0
        # cross entropy
        if label_smoothing:
            eps = 0.1
            log_prob = F.log_softmax(youtputs, dim=2)
            onehot = torch.zeros((youtputs.size(0) * youtputs.size(1), youtputs.size(2))).cuda().scatter(1, ys.view(-1, 1), 1)
            onehot = onehot * (1 - eps) + (1 - onehot) * eps / (youtputs.size(2) - 1)
            onehot = onehot.view(youtputs.size(0), youtputs.size(1), youtputs.size(2))
            for i, t in enumerate(text_lengths):
                loss += -(onehot[i, :t-1, :] * log_prob[i, :t-1, :]).sum() / (t-1)
            loss /= batch_size 
        else:
            loss = F.cross_entropy(youtputs.view(-1, youtputs.size(-1)), ys, ignore_index=trg_pad)
            #loss /= BATCH_SIZE

        print('loss =', loss.item())

        # calc
        n_correct = 0
        for i, t in enumerate(text_lengths):
            tmp = youtputs[i, :t-1, :].max(1)[1].cpu().numpy()
            for j in range(t-1):
                if tmp[j] == text[i][j+1]:
                    n_correct = n_correct + 1
        acc = 1.0 * n_correct / float(sum(text_lengths))

        print('batch size = {}'.format(batch_size))
        print('acc = {}'.format(acc))
        print('lr = {}'.format(lr))
        print('step {}'.format(step))
        step += 1

        sys.stdout.flush()
        # backward
        #print('loss =', loss.item())
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        #reporter.report(verbose=True)
        #torch.cuda.empty_cache()
        #scheduler.step()
    if epoch > 20:
        torch.save(model.state_dict(), save_dir+"/network.epoch{}".format(epoch+1))
        torch.save(optimizer.state_dict(), save_dir+"/network.optimizer.epoch{}".format(epoch+1))
    print('loss =', loss.item())
    print("EPOCH {} end".format(epoch+1))
    #torch.cuda.empty_cache()
