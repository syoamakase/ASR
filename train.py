# -*- coding: utf-8 -*-
import argparse
import copy
import glob
import itertools
import numpy as np
import os
from scipy import fromstring, int16
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import wave

#import hparams as hp
from utils import hparams as hp
from Models.AttModel import AttModel
from Models.CTCModel import CTCModel
from utils.utils import frame_stacking, onehot, load_dat, log_config, load_model, init_weight, adjust_learning_rate, spec_aug
from Loss.label_smoothing import label_smoothing_loss
from legacy.model import Model

import utils_specaug


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_loop(model, optimizer, train_set, scheduler=None):
    num_mb = len(train_set) // hp.batch_size

    if scheduler:
        scheduler.step(epoch)

    for i in range(num_mb):
        # input lmfb (B x T x (F x frame_stacking))
        xs = []
        # target symbols
        ts = []
        # onehot vector of target symbols (B x L x NUM_CLASSES)
        ts_onehot = []
        # vector of target symbols for label smoothing (B x L x NUM_CLASSES)
        ts_onehot_LS = []
        for j in range(hp.batch_size):
            s = train_set[i*hp.batch_size+j].strip()
            x_file, laborg = s.split(' ', 1)
            if '.htk' in x_file:
                cpudat = load_dat(x_file)
                cpudat = cpudat[:, :hp.lmfb_dim]
            elif '.npy'in x_file:
                cpudat = np.load(x_file)
            elif '.wav' in x_file:
                with wave.open(x_file) as wf:
                    dat = wf.readframes(wf.getnframes())
                    y = fromstring(dat, dtype=int16)[:, np.newaxis]
                    y_float = y.astype(np.float32)
                    cpudat = (y_float - np.mean(y_float)) / np.std(y_float)

            #cpudat = spec_aug(cpudat)
            cpudat = torch.from_numpy(cpudat)
            if hp.use_spec_aug:
                T = min(cpudat.shape[1] // 2 - 1, 40)
                #x_norm[i] = utils.time_mask(utils.freq_mask(utils.time_warp(x_norm[i].clone().unsqueeze(0).transpose(1, 2)), num_masks=2), T=T, num_masks=2).transpose(1,2).squeeze(0)
                cpudat = utils_specaug.time_mask(utils_specaug.freq_mask(cpudat.clone().unsqueeze(0).transpose(1, 2), num_masks=2), T=T, num_masks=2).transpose(1,2).squeeze(0).numpy()

            if hp.debug_mode == 'print':
                print("{} {}".format(x_file, cpudat.shape[0]))

            if hp.frame_stacking > 1 and hp.encoder_type != 'Wave':
                cpudat = frame_stacking(cpudat, hp.frame_stacking)

            newlen = cpudat.shape[0]
            if hp.encoder_type == 'CNN':
                cpudat_split = np.split(cpudat, 3, axis = 1)
                cpudat = np.hstack((cpudat_split[0].reshape(newlen, 1, 80),
                            cpudat_split[1].reshape(newlen, 1, 80), cpudat_split[2].reshape(newlen, 1, 80)))

            newlen = cpudat.shape[0]
            #cpulab = np.array([int(i) for i in laborg.split(' ')], dtype=np.int32)
            lab_seq = torch.tensor([int(i) for i in laborg.split(' ')], device=DEVICE).long()
            xs.append(torch.tensor(cpudat, device=DEVICE).float())
            ts.append(lab_seq)

            lab_seq_onehot = F.one_hot(lab_seq, hp.num_classes).float()
            ts_onehot.append(lab_seq_onehot)
            ts_onehot_LS.append(0.9 * lab_seq_onehot + 0.1 * 1.0 / hp.num_classes)

        # to make function
        xs_lengths = torch.tensor(np.array([len(x) for x in xs], dtype=np.int32), device = DEVICE)
        ts_lengths = torch.tensor(np.array([len(t) for t in ts], dtype=np.int32), dtype=torch.float, device = DEVICE)

        padded_xs = nn.utils.rnn.pad_sequence(xs, batch_first = True) 
        padded_ts = nn.utils.rnn.pad_sequence(ts, batch_first = True)
        padded_ts_onehot = nn.utils.rnn.pad_sequence(ts_onehot, batch_first = True)
        padded_ts_onehot_LS = nn.utils.rnn.pad_sequence(ts_onehot_LS, batch_first = True)

        sorted_xs_lengths, perm_index = xs_lengths.sort(0, descending = True)
        sorted_ts_lengths = ts_lengths[perm_index]
        padded_sorted_xs = padded_xs[perm_index] 
        padded_sorted_ts = padded_ts[perm_index]
        padded_sorted_ts_onehot = padded_ts_onehot[perm_index] 
        padded_sorted_ts_onehot_LS = padded_ts_onehot_LS[perm_index]

        #xs, lengths, ts, ts_onehot, ts_onehot_LS, ts_lengths = sort_pad(xs, lengths, ts, ts_onehot, ts_onehot_LS, ts_lengths)

        if hp.output_mode == 'onehot':
            predict_ts = model(padded_sorted_xs, sorted_xs_lengths, padded_sorted_ts_onehot)
        else:
            predict_ts = model(padded_sorted_xs, sorted_xs_lengths, padded_sorted_ts)
            
        loss = -1.0
        if hp.decoder_type == 'Attention':
            for k in range(hp.batch_size):
                num_labels = int(sorted_ts_lengths[k].item())
                loss += label_smoothing_loss(predict_ts[k, :num_labels], padded_sorted_ts_onehot_LS[k, :num_labels]) / num_labels
        elif hp.decoder_type == 'CTC':
            predict_ts = F.log_softmax(predict_ts, dim=2).transpose(0, 1)
            loss = F.ctc_loss(predict_ts, padded_sorted_ts, sorted_xs_lengths, sorted_ts_lengths, blank=hp.num_classes)

        if hp.debug_mode == 'visdom':
            viz.line(X=np.array([i]), Y=np.array([loss.item()]), win='loss', name='train_loss', update='append')
        else:
            print('loss = {}'.format(loss.item()))

        sys.stdout.flush()
        optimizer.zero_grad()
        # backward
        loss.backward()
        clip = 2.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # optimizer update
        optimizer.step()
        loss.detach()
        torch.cuda.empty_cache()

def train_epoch(model, optimizer, train_set, scheduler=None, start_epoch=0):
    for epoch in range(start_epoch, hp.max_epoch):
        train_loop(model, optimizer, train_set, scheduler)
        torch.save(model.state_dict(), hp.save_dir+"/network.epoch{}".format(epoch+1))
        torch.save(optimizer.state_dict(), hp.save_dir+"/network.optimizer.epoch{}".format(epoch+1))
        adjust_learning_rate(optimizer, epoch+1)
        print("EPOCH {} end".format(epoch+1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py')
    args = parser.parse_args()
    
    #overwrite_hparams(args)
    hp.configure(args.hp_file)

    try:
        from visdom import Visdom
        viz = Visdom()
    except:
        if hp.debug_mode == 'visdom':
            raise ModuleNotFoundError

    log_config()
    if hp.legacy:
        model = Model()
    else:
        if hp.decoder_type == 'Attention':
            model = AttModel()
        elif hp.decoder_type == 'CTC':
            model = CTCModel()
     
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
