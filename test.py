# -*- coding: utf-8 
mean_file = '/n/rd23/ueno/e2e/data/aps_sps/nv_wave_mean_var/mean.npy'
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
import sentencepiece as spm
from utils import hparams as hp
from Models.AttModel import AttModel
from Models.CTCModel import CTCModel
from Models.LM import Model_lm
from utils.utils import frame_stacking_legacy, onehot, load_dat, log_config, load_model, overwrite_hparams
from legacy.model import Model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_loop(model, test_set, model_lm):
    batch_size = 1
    word_id = []

    if hp.spm_model is not None:
        sp = spm.SentencePieceProcessor()
        sp.Load(hp.spm_model)

    if hp.word_file:
        with open(hp.word_file) as f:
            for line in f:
                word, _ = line.split(' ', 1)
                word_id.append(word)
    if hasattr(hp, 'mean_file') and hasattr(hp, 'var_file'):
        if hp.mean_file is not None and hp.var_file is not None:
            mean_value = np.load(hp.mean_file).reshape(-1, hp.lmfb_dim)
            var_value = np.load(hp.var_file).reshape(-1, hp.lmfb_dim)
        else:
            mean_value = 0
            var_value = 1
    else:
        mean_value = 0
        var_value = 1

    for i in range(len(test_set)):
        xs = []
        for j in range(batch_size):
            s = test_set[i*batch_size+j].strip()
            x_file = s.strip()
            if '.htk' in x_file:
                cpudat = load_dat(x_file)
                cpudat = cpudat[:, :hp.lmfb_dim]
            elif '.npy' in x_file:
                cpudat = np.load(x_file)
            cpudat -= mean_value

            cpudat /= var_value

            print('{}'.format(x_file), end=' ')
            if hp.frame_stacking and hp.frame_stacking != 1:
                cpudat = frame_stacking_legacy(cpudat, hp.frame_stacking)

            xs.append(torch.tensor(cpudat, device=DEVICE).float())

        # xs, lengths = sort_pad(xs, lengths)
        xs_lengths = torch.tensor(np.array([len(x) for x in xs], dtype=np.int32), device = DEVICE)
        padded_xs = nn.utils.rnn.pad_sequence(xs, batch_first = True)
        sorted_xs_lengths, perm_index = xs_lengths.sort(0, descending = True)
        padded_sorted_xs = padded_xs[perm_index] 

        results = model.decode(padded_sorted_xs, sorted_xs_lengths, model_lm)
        if hp.spm_model:
            print(sp.DecodeIds(results), end='')
        else:
            for character in results:
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
    parser.add_argument('--load_name_lm', type=str, default=None)
    parser.add_argument('--lm_weight', type=float, default=None)
    
    args = parser.parse_args()
    load_name = args.load_name

    load_dir = os.path.dirname(load_name)
    if os.path.exists(os.path.join(load_dir, 'hparams.py')):
        args.hp_file = os.path.join(load_dir, 'hparams.py')

    hp.configure(args.hp_file)
    overwrite_hparams(args)

    if hp.legacy:
        model = Model().to(DEVICE)
    else:
        if hp.decoder_type == 'Attention':
            model = AttModel()
        elif hp.decoder_type == 'CTC':
            model = CTCModel()

    if args.load_name_lm is not None:
        model_lm = Model_lm()
        model_lm.to(DEVICE)
        model_lm.load_state_dict(load_model(args.load_name_lm))
        model_lm.eval()
    else:
        model_lm = None

    model = model.to(DEVICE)
    model.eval()

    test_set = []
    with open(hp.test_script) as f:
        for line in f:
            filename = line.split(' ')[0]
            test_set.append(filename)

    model.load_state_dict(load_model(load_name))
    test_loop(model, test_set, model_lm)
