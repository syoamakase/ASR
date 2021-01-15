# -*- coding: utf-8 -*-
import argparse
import copy
import glob
import itertools
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import hparams as hp
from Models.LM import Model_lm
from utils.utils import frame_stacking, onehot, load_dat, log_config, load_model, init_weight, adjust_learning_rate, spec_aug, fill_variables
from Loss.label_smoothing import label_smoothing_loss
import datasets_LM

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_loop(model, optimizer, epoch):
    dataset_train = datasets_LM.get_dataset(hp.train_script_lm)
    dataloader = DataLoader(dataset_train, batch_size=hp.batch_size_LM, shuffle=True, num_workers=2, collate_fn=datasets_LM.collate_fn, pin_memory=True, drop_last=True)
    #pbar = tqdm(dataloader)
    #for d in pbar:
    step = 0
    len_train = len(dataloader)
    for d in dataloader:
        step += 1
        text, pos_text, text_lengths = d

        text = text.to(DEVICE)
        pos_text = pos_text.to(DEVICE)
        text_lengths = text_lengths.to(DEVICE)

        predict_ts = model(text[:,:-1])
        
        loss = label_smoothing_loss(predict_ts, text[:, 1:].contiguous(), text_lengths, hp.T_norm, hp.B_norm)

        optimizer.zero_grad()
        # backward
        loss.backward()
        clip = 5.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # optimizer update
        optimizer.step()
        loss.detach()
        if hp.debug_mode == 'tensorboard':
            writer.add_scalar("Loss/train_lm", loss, epoch*len_train+step)
        else:
            print('loss = {}'.format(loss.item()))

        sys.stdout.flush()

def train_epoch(model, optimizer, start_epoch=0):
    for epoch in range(start_epoch, hp.max_epoch):
        start_time = time.time()
        train_loop(model, optimizer, epoch)
        if (epoch + 1) % hp.save_per_epoch == 0 or (epoch+1) % hp.reset_optimizer_epoch > 30:
            torch.save(model.state_dict(), hp.save_dir+"/network.epoch{}".format(epoch+1))
            torch.save(optimizer.state_dict(), hp.save_dir+"/network.optimizer.epoch{}".format(epoch+1))
        adjust_learning_rate(optimizer, epoch+1)
        if (epoch+1) % hp.reset_optimizer_epoch == 0:
            optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
        print("EPOCH {} end".format(epoch+1))
        print(f'elapsed time = {(time.time()-start_time)//60}m')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py')
    args = parser.parse_args()
    
    #overwrite_hparams(args)
    hp.configure(args.hp_file)
    fill_variables()
    hp.save_dir = os.path.join(hp.save_dir, 'LM')
    os.makedirs(hp.save_dir, exist_ok=True)

    if hp.debug_mode == 'tensorboard':
        writer = SummaryWriter(f'{hp.save_dir}/logs/{hp.comment}')

    log_config()
    model = Model_lm(hp)
     
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
        if hp.is_flatstart:
            load_epoch = 0
        else:
            optimizer.load_state_dict(torch.load(os.path.join(hp.load_checkpoints_path, 'network.optimizer.epoch{}'.format(load_epoch))))

    train_epoch(model, optimizer, start_epoch=load_epoch)
