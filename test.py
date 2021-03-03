# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import sys
import sentencepiece as spm
import torch
import torch.nn as nn

from Models.AttModel import AttModel
from Models.CTCModel import CTCModel
from Models.LM import Model_lm
from utils import hparams as hp
import utils
from utils.utils import frame_stacking, load_dat, log_config, load_model, overwrite_hparams, fill_variables
from text import text_to_sequence, sequence_to_text

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_loop(model, test_set, model_lm):
    batch_size = 1

    if hp.spm_model is not None:
        sp = spm.SentencePieceProcessor()
        sp.Load(hp.spm_model)

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
            s = test_set[i * batch_size + j].strip()
            x_file = s.strip()
            if '.htk' in x_file:
                cpudat = load_dat(x_file)
                cpudat = cpudat[:, :hp.lmfb_dim]
            elif '.npy' in x_file:
                cpudat = np.load(x_file)
            cpudat -= mean_value
            cpudat /= np.sqrt(var_value)

            print('{}'.format(x_file), end=' ')

            xs.append(torch.tensor(cpudat, device=DEVICE).float())

        xs_lengths = torch.tensor([len(x) for x in xs]).long()
        padded_xs = nn.utils.rnn.pad_sequence(xs, batch_first=True)
        sorted_xs_lengths, perm_index = xs_lengths.sort(0, descending=True)
        padded_sorted_xs = padded_xs[perm_index]

        padded_sorted_xs, sorted_xs_lengths = frame_stacking(padded_sorted_xs, sorted_xs_lengths, hp.frame_stacking)
        pos_mel = torch.arange(1, sorted_xs_lengths[0] + 1).to(DEVICE).unsqueeze(0)

        results = model.decode_v2(padded_sorted_xs, sorted_xs_lengths, model_lm, pos_mel)
        #results = model.decode(padded_sorted_xs, sorted_xs_lengths, model_lm, pos_mel)
        if hp.spm_model:
            print(sp.DecodeIds(results), end='')
        else:
            for character in results:
                print(character, end=' ')
                #print(sequence_to_text(character), end='')
        print()
        sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_name')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py')
    parser.add_argument('--test_script', type=str, default=None)
    parser.add_argument('--load_name_lm', type=str, default=None)
    parser.add_argument('--lm_weight', type=float, default=None)
    parser.add_argument('--log_params', action='store_true')
    
    args = parser.parse_args()
    load_name = args.load_name

    load_dir = os.path.dirname(load_name)
    if os.path.exists(os.path.join(load_dir, 'hparams.py')):
        args.hp_file = os.path.join(load_dir, 'hparams.py')

    #hp = utils.HParams()
    hp.configure(args.hp_file)

    fill_variables(hp, args.log_params)
    overwrite_hparams(args, hp)

    if hp.decoder_type == 'Attention':
        model = AttModel(hp)
    elif hp.decoder_type == 'CTC':
        model = CTCModel(hp)

    if hp.load_name_lm is not None:
        hp_LM_path = os.path.join(os.path.dirname(hp.load_name_lm), 'hparams.py')
        hp_LM = utils.HParams()
        hp_LM.configure(hp_LM_path)
        model_lm = Model_lm(hp_LM)
        model_lm.to(DEVICE)
        model_lm.load_state_dict(load_model(hp.load_name_lm))
        model_lm.eval()
    else:
        model_lm = None

    model = model.to(DEVICE)
    model.eval()

    test_set = []
    with open(hp.test_script) as f:
        for line in f:
            filename = line.split(' ')[0]
            filename = filename.split('|')[0]
            test_set.append(filename)

    model.load_state_dict(load_model(load_name))
    test_loop(model, test_set, model_lm)
