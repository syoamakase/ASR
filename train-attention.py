# -*- coding: utf-8 -*-
"""
19/09/2018
Implementation of attention model using torch==1.0.0a0+8601b33

This script applies Label Smoothing.

Update:
    multi-gpu
"""

import copy
import itertools
import os
from struct import unpack, pack
import sys

import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

FRAME_STACKING = True
NUM_HIDDEN_NODES = 320
NUM_LAYERS = int(sys.argv[1]) # the number of bilstm layer
BATCH_SIZE = int(sys.argv[2]) # 30
NUM_CLASSES = int(sys.argv[3]) # 3260 / 34331
ATTENTION_OPTION = sys.argv[4] # "tanh" or "conv"
script_file = sys.argv[5]
save_dir = sys.argv[6] # save dir name

#torch.manual_seed(7)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('NUM_LAYERS =', NUM_LAYERS)
print('BATCH_SIZE =', BATCH_SIZE)
print('NUM_CLASSES =', NUM_CLASSES)
print('ATTENTION_OPTION =', ATTENTION_OPTION)
print('script_file =', script_file)
print('save_dir =', save_dir)

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
def sort_pad(xs, ts, ts_onehot, ts_onehot_LS, lengths, ts_lengths):
    """
    To sort "lengths" order.
    This funtion is needed for "torch.nn.utils.rnn.pack_padded_sequence()"

    Args:
        xs : input feature. (BATCH SIZE, time frames, log mel-scale filter bank)
        ts : grand truth data (BATCH SIZE, label lengths)
        ts_onehot : grand truth data which is an onehot vector (BATCH SIZE, label lengths, #labels)
        ts_onehot_LS : grand truth data which is a vector for label smoothing (BATCH SIZE, label lengths, #labels)
        lengths : the lengths of the input feature (BATCH SIZE)
        ts_lengths : the lengths of grand truth data (BATCH SIZE)

    Returns:
        xs_tensor : "torch FloatTensor" of sorted xs
        ts_results : list of sorted ts
        ts_onehot_tensor : "torch FloatTensor" of sorted ts_onehot
        lengths : sorted lenghts
    """
    arg_lengths = argsort(lengths)
    maxlen = max(lengths)
    ts_maxlen = max(ts_lengths)
    xs_tensor = torch.zeros((BATCH_SIZE, maxlen, 120), dtype=torch.float32, requires_grad=True).to(DEVICE)
    ts_onehot_tensor = torch.zeros((BATCH_SIZE, ts_maxlen, NUM_CLASSES), dtype=torch.float32, requires_grad=True).to(DEVICE)
    ts_onehot_LS_tensor = torch.zeros((BATCH_SIZE, ts_maxlen, NUM_CLASSES), dtype=torch.float32, requires_grad=True).to(DEVICE)
    lengths_tensor = torch.zeros((BATCH_SIZE), dtype=torch.int64).to(DEVICE)
    ts_result = []
    
    for i, i_sort in enumerate(arg_lengths):
        xs_tensor.data[i, 0:lengths[i_sort]] = torch.from_numpy(xs[i_sort])
        ts_onehot_tensor.data[i, 0:ts_lengths[i_sort]] = torch.from_numpy(ts_onehot[i_sort])
        ts_onehot_LS_tensor.data[i, 0:ts_lengths[i_sort]] = torch.from_numpy(ts_onehot_LS[i_sort])
        ts_result.append(Variable(torch.from_numpy(ts[i_sort]).to(DEVICE).long()))
        lengths_tensor.data[i] = lengths[i_sort] 

    return xs_tensor, ts_result, ts_onehot_tensor, ts_onehot_LS_tensor, lengths_tensor


def init_weight(m):
    """
    To initialize weights and biases.
    """
    classname = m.__class__.__name__
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


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # encoder
        self.bi_lstm = nn.LSTM(input_size=120, hidden_size=NUM_HIDDEN_NODES, num_layers=NUM_LAYERS, batch_first=True, dropout=0.2, bidirectional=True)
        # attention
        self.L_se = nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES * 2, bias=False)
        self.L_he = nn.Linear(NUM_HIDDEN_NODES * 2, NUM_HIDDEN_NODES * 2)
        self.L_ee = nn.Linear(NUM_HIDDEN_NODES * 2, 1, bias=False)
        # conv attention
        self.F_conv1d = nn.Conv1d(1, 10, 100, stride=1, padding=50, bias=False)
        self.L_fe = nn.Linear(10, NUM_HIDDEN_NODES * 2, bias=False)

        # decoder
        self.L_sy = nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES, bias=False)
        self.L_gy = nn.Linear(NUM_HIDDEN_NODES * 2, NUM_HIDDEN_NODES)
        self.L_yy = nn.Linear(NUM_HIDDEN_NODES, NUM_CLASSES)

        self.L_ys = nn.Linear(NUM_CLASSES, NUM_HIDDEN_NODES * 4 , bias=False)
        self.L_ss = nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES * 4, bias=False)
        self.L_gs = nn.Linear(NUM_HIDDEN_NODES * 2, NUM_HIDDEN_NODES * 4)

    def forward(self, x, lengths, target):
        total_length = x.size(1)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        
        #self.bi_lstm.flatten_parameters()
        h, _ = self.bi_lstm(x)

        hbatch, lengths = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=total_length)
        # hbatch : BATCH_SIZE x num_frames x NUM_HIDDEN_NODES
        # lengths : BATCH_SIZE lengths array
        # target:  BATCH_SIZE x **num_labels** x NUM_CLASSES
        batch_size = hbatch.size(0)
        num_frames = hbatch.size(1)
        num_labels = target.size(1)

        e_mask = torch.ones((batch_size, num_frames, 1), device=DEVICE, requires_grad=False)

        s = torch.zeros((batch_size, NUM_HIDDEN_NODES), device=DEVICE, requires_grad=False)
        c = torch.zeros((batch_size, NUM_HIDDEN_NODES), device=DEVICE, requires_grad=False)

        youtput = torch.zeros((batch_size, num_labels, NUM_CLASSES), device=DEVICE, requires_grad=False)
        alpha = torch.zeros((batch_size, 1,  num_frames), device=DEVICE, requires_grad=False)

        for i, tmp in enumerate(lengths):
            if tmp < num_frames:
                e_mask.data[i, tmp:] = 0.0

        for step in range(num_labels):
            if ATTENTION_OPTION == "conv":
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
            elif ATTENTION_OPTION == 'tanh':
                # e : B x T x 2H
                # se(s) : B x 2H -> se(s).unsqueeze(1): B x T x 2H
                # he(hbatch) : B x T x 2H
                e = torch.tanh(self.L_se(s).unsqueeze(1) + self.L_he(hbatch))
                # ee(e) : BATCH_SIZE x num_frames x 1 -> ee(e).squeeze(2) : BATCH_SIZE x num_frames
                # that is, new e : BATCH_SIZE x num_frames
                e = self.L_ee(e).squeeze(2)

            # e_nonlin : BATCH_SIZE x num_frames
            #e_nonlin = e.exp()
            e_nonlin = (e - e.max(1)[0].unsqueeze(1)).exp()
            # e_nonlin : BATCH_SIZE x num_frames
            e_nonlin = e_nonlin * e_mask

            alpha = e_nonlin / e_nonlin.sum(dim=1, keepdim=True)
            g = (alpha * hbatch).sum(dim=1)
            alpha = alpha.transpose(1, 2)

            # generate
            y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s)))

            # recurrency calcuate
            rec_input = self.L_ys(target[:,step,:]) + self.L_ss(s) + self.L_gs(g)
            batch = rec_input.size(0)

            # LSTM like calcuation.
            # F.lstm() in chainer is the same
            ingate, forgetgate, cellgate, outgate = rec_input.chunk(4, 1)
            half = 0.5
            ingate = torch.tanh(ingate * half) * half + half
            forgetgate = torch.tanh(forgetgate * half) * half + half
            cellgate = torch.tanh(cellgate)
            outgate = torch.tanh(outgate * half) * half + half
            c_next = (forgetgate * c) + (ingate * cellgate)
            h = outgate * torch.tanh(c_next)
            s = h
            c = c_next

            youtput[:,step] = y

        return youtput

# initialize pytorch
model = Model()
model.apply(init_weight)

# multi-gpu setup
if torch.cuda.device_count() > 1:
    # multi-gpu configuration
    ngpu = torch.cuda.device_count()
    device_ids = list(range(ngpu))
    model = torch.nn.DataParallel(model, device_ids)
    model.cuda()
else:
    model.to(DEVICE)

# 0.4 -> 1.0: 
# criterion = nn.CrossEntropyLoss(size_average=True)
criterion = nn.CrossEntropyLoss(reduction='elementwise_mean')

# model.eval()
model.train()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

os.makedirs(save_dir, exist_ok=True)

print('PID =', os.getpid())
print('HOST =', os.uname()[1])
print('Weight decay = ', optimizer.param_groups[0]['weight_decay'])

# load
#model.load_state_dict(load_model("model.word.mb40_LS-SS_fix/network.epoch30"))
#optimizer.load_state_dict(torch.load("model.word.mb40_LS-SS_fix/network.optimizer.epoch30"))

for epoch in range(0, 30):
    script_buf = []
    with open(script_file) as f:
        for line in f:
            script_buf.append(line)

    num_mb = len(script_buf) // BATCH_SIZE
    maxlen = 0
    scheduler.step(epoch)
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
            x_file, laborg = s.split(' ', 1)
            if '.htk' in x_file:
                cpudat = load_dat(x_file)
                cpudat = cpudat[:, :40]
            elif '.npy'in x_file:
                cpudat = np.load(x_file)

            print("{} {}".format(x_file, cpudat.shape[0]))
            if FRAME_STACKING:
                newlen = int(cpudat.shape[0] / 3)
                cpudat = cpudat[:3 * newlen, :]
                cpudat = np.reshape(cpudat, (newlen, 3, 40))
                cpudat = np.reshape(cpudat, (newlen, 3 * 40)).astype(np.float32)
            lengths.append(newlen)
            xs.append(cpudat)
            cpulab = np.array([int(i) for i in laborg.split(' ')], dtype=np.int32)
            cpulab_onehot = onehot(cpulab, NUM_CLASSES)
            ts.append(cpulab)
            ts_lengths.append(len(cpulab))
            ts_onehot.append(cpulab_onehot)
            ts_onehot_LS.append(0.9 * onehot(cpulab, NUM_CLASSES) + 0.1 * 1.0 / NUM_CLASSES)

        xs, ts, ts_onehot, ts_onehot_LS, lengths = sort_pad(xs, ts, ts_onehot, ts_onehot_LS, lengths, ts_lengths)

        youtput_in_Variable = model(xs, lengths, ts_onehot)

        loss = 0.0
        for i in range(BATCH_SIZE):
            num_labels = ts[i].size(0)
            loss += -(F.log_softmax(youtput_in_Variable[i][:num_labels], dim=1) * ts_onehot_LS[i][:num_labels]).sum() / num_labels

        print('loss =', loss.item())
        sys.stdout.flush()
        optimizer.zero_grad()
        # backward
        loss.backward()
        clip = 5.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # optimizer update
        optimizer.step()
        loss.detach()
    torch.save(model.state_dict(), save_dir+"/network.epoch{}".format(epoch+1))
    torch.save(optimizer.state_dict(), save_dir+"/network.optimizer.epoch{}".format(epoch+1))
    print("EPOCH {} end".format(epoch))
