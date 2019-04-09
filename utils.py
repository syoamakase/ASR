# -*- coding: utf-8 -*-
import numpy as np
from struct import unpack, pack
import sys
import torch

import hparams as hp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_config():
    for key in hp.__dict__.keys():
        if not '__' in key:
            print('{} {}'.format(key, eval(hp+'.'+key)))

def load_dat(filename):
    """
    To read binary data in htk file.
    The htk file includes log mel-scale filter bank.

    Args:
        filename : file name to read htk file

    Returns:
        dat : 120 (means log mel-scale filter bank) x T (time frame)

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

def frame_stacking(cpudat, stack):
    newlen = int(cpudat.shape[0] / stack)
    cpudat = cpudat[:stack * newlen, :]
    cpudat = np.reshape(cpudat, (newlen, stack, hp.lmfb_dim))
    cpudat = np.reshape(cpudat, (newlen, stack * hp.lmfb_dim)).astype(np.float32)
    return cpudat, newlen

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

# sorting lengths order
def sort_pad(xs, lengths, ts=None, ts_onehot=None, ts_onehot_LS=None, ts_lengths=None):
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
    def argsort(seq):
        return np.argsort(np.array(seq))[::-1].tolist()
    arg_lengths = argsort(lengths)
    maxlen = max(lengths)
    
    input_size = hp.lmfb_dim * hp.frame_stacking if frame_stacking else hp.lmfb_dim
 
    if (ts is not None) and (ts_lengths is not None) and (ts_onehot_LS is not None):
        xs_tensor = torch.zeros((hp.batch_size, maxlen, input_size), dtype=torch.float32, device=DEVICE, requires_grad=True)
        ts_maxlen = max(ts_lengths)
        ts_onehot_tensor = torch.zeros((hp.batch_size, ts_maxlen, hp.num_classes), dtype=torch.float32, device=DEVICE, requires_grad=True)
        ts_onehot_LS_tensor = torch.zeros((hp.batch_size, ts_maxlen, hp.num_classes), dtype=torch.float32, device=DEVICE, requires_grad=True)
        lengths_tensor = torch.zeros((hp.batch_size), dtype=torch.int64, device=DEVICE)
        ts_result = []
        for i, i_sort in enumerate(arg_lengths):
            xs_tensor.data[i, 0:lengths[i_sort]] = torch.from_numpy(xs[i_sort])
            ts_onehot_tensor.data[i, 0:ts_lengths[i_sort]] = torch.from_numpy(ts_onehot[i_sort])
            ts_onehot_LS_tensor.data[i, 0:ts_lengths[i_sort]] = torch.from_numpy(ts_onehot_LS[i_sort])
            ts_result.append(torch.tensor(ts[i_sort], dtype=torch.long, device=DEVICE))
            lengths_tensor.data[i] = lengths[i_sort] 
        return xs_tensor, lengths_tensor, ts_result, ts_onehot_tensor, ts_onehot_LS_tensor
    else:
        xs_tensor = torch.zeros((1, maxlen, input_size), dtype=torch.float32, device=DEVICE, requires_grad=True)
        lengths_tensor = torch.zeros((1), dtype=torch.int64, device=DEVICE)
        for i, i_sort in enumerate(arg_lengths):
            xs_tensor.data[i, 0:lengths[i_sort]] = torch.from_numpy(xs[i_sort])
            lengths_tensor.data[i] = lengths[i_sort] 
        return  xs_tensor, lengths_tensor

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