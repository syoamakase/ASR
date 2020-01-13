# -*- coding: utf-8 -*-
import argparse
import copy
import itertools
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

#import hparams as hp
from utils import hparams as hp
from Models.AttModel import AttModel
from Models.CTCModel import CTCModel
from utils.utils import frame_stacking, onehot, load_dat, log_config, load_model, overwrite_hparams
from legacy.model import Model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_loop(model, test_set):
    batch_size = 1
    word_id = []
    if hp.word_file:
        with open(hp.word_file) as f:
            for line in f:
                word, _ = line.split(' ', 1)
                word_id.append(word)

    for i in range(len(test_set)):
        xs = []
        for j in range(batch_size):
            s = test_set[i*batch_size+j].strip()
            x_file = s.strip()
            if '.htk' in x_file:
                cpudat = load_dat(x_file)
                cpudat = cpudat[:, :hp.lmfb_dim]
            elif '.npy'in x_file:
                cpudat = np.load(x_file)

            print('{}'.format(x_file), end=' ')
            if hp.frame_stacking and hp.frame_stacking != 1:
                cpudat = frame_stacking(cpudat, hp.frame_stacking)

            xs.append(torch.tensor(cpudat, device=DEVICE).float())

        # xs, lengths = sort_pad(xs, lengths)
        xs_lengths = torch.tensor(np.array([len(x) for x in xs], dtype=np.int32), device = DEVICE)
        padded_xs = nn.utils.rnn.pad_sequence(xs, batch_first = True)
        sorted_xs_lengths, perm_index = xs_lengths.sort(0, descending = True)
        padded_sorted_xs = padded_xs[perm_index] 

        results = model.decode(padded_sorted_xs, sorted_xs_lengths)
        for character in results:
            if hp.word_file:
                if word_id[character] != '<sos>' and word_id[character] != '<eos>':
                    print(word_id[character], end='')
            else:
                print(character, end=' ')
        print()
        sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--load_model', default=hp.load_checkpoints_path+'/network.epoch{}'.format(hp.load_checkpoints_epoch))
    parser.add_argument('--load_name')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py')
    parser.add_argument('--test_script', type=str, default=None)
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--eos_id', type=int, default=None)
    parser.add_argument('--output_mode', type=str, default=None)
    
    args = parser.parse_args()
    load_name = args.load_name

    hp.configure(args.hp_file)
    overwrite_hparams(args)

    if hp.legacy:
        model = Model().to(DEVICE)
    else:
        if hp.decoder_type == 'Attention':
            model = AttModel()
        elif hp.decoder_type == 'CTC':
            model = CTCModel()
    model = model.to(DEVICE)

    model.eval()

    test_set = []
    with open(hp.test_script) as f:
        for line in f:
            filename = line.split(' ')[0]
            test_set.append(filename)

    model.load_state_dict(load_model(load_name))
    test_loop(model, test_set)
