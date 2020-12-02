# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
from struct import unpack
import torch
import torch.nn as nn

from utils import hparams as hp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log_config():
    print(f'PID = {os.getpid()}')
    print(f'PyTorch version = {torch.__version__}')
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print('cuda device = {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    for key in hp.__dict__.keys():
        if not '__' in key:
            print('{} = {}'.format(key, eval('hp.' + key)))


def load_dat(filename):
    """
    To read binary data in htk file.
    The htk file includes log mel-scale filter bank.

    Args:
        filename : file name to read htk file

    Returns:
        dat : (log mel-scale filter bank dim) x (time frame)

    """
    fh = open(filename, "rb")
    spam = fh.read(12)
    _, _, sampSize, _ = unpack(">IIHH", spam)
    veclen = int(sampSize / 4)
    fh.seek(12, 0)
    dat = np.fromfile(fh, dtype=np.float32)
    dat = dat.reshape(int(len(dat) / veclen), veclen)
    dat = dat.byteswap()
    fh.close()
    return dat


def frame_stacking(x, x_lengths, stack):
    if stack == 1:
        return x, x_lengths
    else:
        batch_size = x.shape[0]
        newlen = x.shape[1] // stack
        x_lengths = x_lengths // stack
        stacked_x = x[:, 0:newlen * stack].reshape(batch_size, newlen, -1)
        return stacked_x, x_lengths


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
    """
    To load PyTorch models either of single-gpu and multi-gpu based model
    """
    model_state = torch.load(model_file)
    is_multi_loading = True if torch.cuda.device_count() > 1 else False
    # This line may include bugs!!
    is_multi_loaded = True if 'module' in list(model_state.keys())[0] else False

    if is_multi_loaded is is_multi_loading:
        return model_state

    # the model to load is multi-gpu and the model to use is single-gpu
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

def adjust_learning_rate(optimizer, epoch):
    if hp.reset_optimizer_epoch is not None:
        if (epoch % hp.reset_optimizer_epoch) > hp.lr_adjust_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8
    else:
        if epoch > hp.lr_adjust_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8


def get_transformer_learning_rate(step, d_model=256):
    warmup_step = 10000
    warmup_factor = 1.0
    return warmup_factor * min(step ** -0.5, step * warmup_step ** -1.5) * (d_model ** -0.5)

 
def overwrite_hparams(args):
    for key, value in args._get_kwargs():
        if value is not None and value != 'load_name':
            setattr(hp, key, value)


def fill_variables(verbose=True):
    if hasattr(hp, 'num_hidden_nodes'):
        num_hidden_nodes_encoder = hp.num_hidden_nodes
        num_hidden_nodes_decoder = hp.num_hidden_nodes
    else:
        num_hidden_nodes_encoder = 512
        num_hidden_nodes_decoder = 512

    default_var = {'spm_model': None, 'T_norm': True, 'B_norm': False, 'save_per_epoch': 1, 'lr_adjust_epoch': 20,
                   'reset_optimizer_epoch': 40, 'num_hidden_nodes_encoder': num_hidden_nodes_encoder, 'num_hidden_nodes_decoder': num_hidden_nodes_decoder,
                    'comment': '', 'load_name_lm': None, 'shuffle': False, 'num_mask_F': 1, 'num_mask_T': 1, 'clip': 5.0,
                    'max_width_F': 27, 'max_width_T': 100, 'mean_file': None, 'wav_file': None, 'accum_grad': 1}
    for key, value in default_var.items():
        if not hasattr(hp, key):
            if verbose:
                print('{} is not found in hparams. defalut {} is used.'.format(key, value))
            setattr(hp, key, value)
