# -*- coding: utf-8 -*-
import copy
import itertools
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import hparams as hp
from Models.AttModel import AttModel
from utils import frame_stacking, onehot, load_dat, log_config, sort_pad, load_model
from legacy.model import Model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_loop(model, test_set):
    batch_size = 1
    for i in range(len(test_set)):
        xs = []
        lengths = []
        for j in range(batch_size):
            s = test_set[i*batch_size+j].strip()
            x_file = s.strip()
            if '.htk' in x_file:
                cpudat = load_dat(x_file)
                cpudat = cpudat[:, :hp.lmfb_dim]
            elif '.npy'in x_file:
                cpudat = np.load(x_file)

            print('{}'.format(x_file), end=' ')
            if (hp.frame_stacking is not None) or hp.frame_stacking == 1:
                cpudat, newlen = frame_stacking(cpudat, hp.frame_stacking)
            else:
                newlen = cpudat.shape[0]
            lengths.append(newlen)
            xs.append(cpudat)

        xs, lengths = sort_pad(xs, lengths)
        results = model.decode(xs, lengths)
        for character in results:
            print(character, end=' ')
        print()
        sys.stdout.flush()

if __name__ == "__main__":
    if hp.legacy:
        model = Model().to(DEVICE)
    else:
        model = AttModel().to(DEVICE)

    test_set = []
    with open(hp.test_script) as f:
        for line in f:
            test_set.append(line)
    
    assert hp.load_checkpoints, 'Please specify the checkpoints'

    model.load_state_dict(load_model(os.path.join(hp.load_checkpoints_path, 'network.epoch{}'.format(hp.load_checkpoints_epoch))))
    test_loop(model, test_set)
