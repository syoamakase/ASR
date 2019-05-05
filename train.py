# -*- coding: utf-8 -*-
import copy
import glob
import itertools
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

import hparams as hp
from Models.AttModel import AttModel
from utils import frame_stacking, onehot, load_dat, log_config, sort_pad, load_model, init_weight
from Loss.label_smoothing import label_smoothing_loss
from legacy.model import Model

try:
    from visdom import Visdom
    viz = Visdom()
except:
    if hp.debug_mode == 'visdom':
        raise ModuleNotFoundError

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_loop(model, optimizer, train_set, scheduler=None):
    num_mb = len(train_set) // hp.batch_size

    if scheduler:
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
        for j in range(hp.batch_size):
            s = train_set[i*hp.batch_size+j].strip()
            x_file, laborg = s.split(' ', 1)
            if '.htk' in x_file:
                cpudat = load_dat(x_file)
                cpudat = cpudat[:, :hp.lmfb_dim]
            elif '.npy'in x_file:
                cpudat = np.load(x_file)

            if hp.debug_mode == 'print':
                print("{} {}".format(x_file, cpudat.shape[0]))
            if hp.frame_stacking > 1:
                cpudat, newlen = frame_stacking(cpudat, hp.frame_stacking)

            newlen = cpudat.shape[0]
            if hp.encoder_type == 'CNN':
                cpudat_split = np.split(cpudat, 3, axis = 1)
                cpudat = np.hstack((cpudat_split[0].reshape(newlen, 1, 80),
                            cpudat_split[1].reshape(newlen, 1, 80), cpudat_split[2].reshape(newlen, 1, 80)))
            newlen = cpudat.shape[0]
            lengths.append(newlen)
            xs.append(cpudat)
            cpulab = np.array([int(i) for i in laborg.split(' ')], dtype=np.int32)
            cpulab_onehot = onehot(cpulab, hp.num_classes)
            ts.append(cpulab)
            ts_lengths.append(len(cpulab))
            ts_onehot.append(cpulab_onehot)
            ts_onehot_LS.append(0.9 * onehot(cpulab, hp.num_classes) + 0.1 * 1.0 / hp.num_classes)

        xs, lengths, ts, ts_onehot, ts_onehot_LS = sort_pad(xs, lengths, ts, ts_onehot, ts_onehot_LS, ts_lengths)
        youtput_in_Variable = model(xs, lengths, ts_onehot)

        loss = 0.0
        for k in range(hp.batch_size):
            num_labels = ts[k].size(0)
            loss += label_smoothing_loss(youtput_in_Variable[k][:num_labels], ts_onehot_LS[k][:num_labels]) / num_labels

        if hp.debug_mode == 'visdom':
            viz.line(X=np.array([i]), Y=np.array([loss.item()]), win='loss', name='train_loss', update='append')
        else:
            print('loss = {}'.format(loss.item()))

        sys.stdout.flush()
        optimizer.zero_grad()
        # backward
        loss.backward()
        clip = 5.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # optimizer update
        optimizer.step()
        loss.detach()
        torch.cuda.empty_cache()

def train_epoch(model, optimizer, train_set, scheduler=None, start_epoch=0):
    for epoch in range(start_epoch, hp.max_epoch+1):
        train_loop(model, optimizer, train_set, scheduler)
        torch.save(model.state_dict(), hp.save_dir+"/network.epoch{}".format(epoch+1))
        torch.save(optimizer.state_dict(), hp.save_dir+"/network.optimizer.epoch{}".format(epoch+1))
        print("EPOCH {} end".format(epoch+1))

if __name__ == "__main__":
    log_config()
    if hp.legacy:
        model = Model()
    else:
        model = AttModel()
    
    model.apply(init_weight)

    if torch.cuda.device_count() > 1:
        # multi-gpu configuration
        ngpu = torch.cuda.device_count()
        device_ids = list(range(ngpu))
        model = torch.nn.DataParallel(model, device_ids)
    model.to(DEVICE)        
    
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    os.makedirs(hp.save_dir, exist_ok=True)

    load_epoch = 0
    if hp.load_checkpoints:
        if hp.load_checkpoints_epoch is None:
            path_list = glob.glob(os.path.join(hp.load_checkpoints_path, 'network.epoch*'))
            for path in path_list:
                epoch = int(path.split('.')[-1].replace('epoch', ''))
                if epoch > load_epoch:
                    load_epoch = epoch
        else:
            load_epoch = hp.load_checkpoints_epoch
        print("{} epoch {} load".format(hp.load_checkpoints_path, load_epoch))
        model.load_state_dict(load_model(os.path.join(hp.load_checkpoints_path, 'network.epoch{}'.format(load_epoch))))
        optimizer.load_state_dict(torch.load(os.path.join(hp.load_checkpoints_path, 'network.optimizer.epoch{}'.format(load_epoch))))

    train_set = []
    with open(hp.train_script) as f:
        for line in f:
            train_set.append(line)

    train_epoch(model, optimizer, train_set, start_epoch=load_epoch)
