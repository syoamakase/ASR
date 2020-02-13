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
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import wave
from tqdm import tqdm

#import hparams as hp
from utils import hparams as hp
from Models.AttModel import AttModel
from Models.CTCModel import CTCModel
from utils.utils import frame_stacking, onehot, load_dat, log_config, load_model, init_weight, adjust_learning_rate, spec_aug, save_hparams
from Loss.label_smoothing import label_smoothing_loss
from legacy.model import Model
import Datasets

import utils_specaug

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_loop(model, optimizer, train_set, scheduler=None):
    datasets = Datasets.get_dataset(hp.train_script)
    dataloader = DataLoader(datasets, batch_size=40, num_workers=4, collate_fn=Datasets.collate_fn)
    pbar = tqdm(dataloader)
    for d in pbar:
        if scheduler:
            scheduler.step(epoch)

        text, mel_input, pos_text, pos_mel, text_lengths, mel_lengths = d

        if hp.use_spec_aug:
            T = min(mel_input.shape[1] // 2 - 1, 40)
            mel_input = utils_specaug.time_mask(utils_specaug.freq_mask(mel_input.clone().unsqueeze(0).transpose(1, 2), num_masks=2), T=T, num_masks=2).transpose(1,2).squeeze(0)

        text = text.to(DEVICE)
        mel_input = mel_input.to(DEVICE)
        pos_text = pos_text.to(DEVICE)
        pos_mel = pos_mel.to(DEVICE)
        text_lengths = text_lengths.to(DEVICE)

        text_input = text[:, :-1]

        if hp.frame_stacking > 1 and hp.encoder_type != 'Wave':
            mel_input, mel_lengths = frame_stacking(mel_input, mel_lengths, hp.frame_stacking)

        predict_ts = model(mel_input, mel_lengths, text_input)
        
        if hp.decoder_type == 'Attention':
            target = text[:, 1:].contiguous().view(-1)
            loss = label_smoothing_loss(predict_ts, target, text_lengths)
        elif hp.decoder_type == 'CTC':
            predict_ts = F.log_softmax(predict_ts, dim=2).transpose(0, 1)
            loss = F.ctc_loss(predict_ts, text, mel_lengths, text_lengths, blank=hp.num_classes)

        if hp.debug_mode == 'visdom':
            viz.line(X=np.array([i]), Y=np.array([loss.item()]), win='loss', name='train_loss', update='append')
        else:
            print('loss = {}'.format(loss.item()))

        sys.stdout.flush()
        optimizer.zero_grad()
        # backward
        loss.backward()
        clip = 1.0
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
    os.makedirs(hp.save_dir, exist_ok=True)
    save_hparams(args.hp_file, os.path.join(hp.save_dir, os.path.basename(args.hp_file)))

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
