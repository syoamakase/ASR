import os
import sys
import time
import numpy as np
import random
from struct import unpack, pack

import argparse

np.seterr(all='warn')

parser = argparse.ArgumentParser()
parser.add_argument('-S', '--script_filename', required=True)
parser.add_argument('-C', '--calc_normalize', action='store_true')
parser.add_argument('--ext', required=True)
args = parser.parse_args()
script_filename = args.script_filename
calc_normalize = args.calc_normalize
ext = args.ext

assert ext == 'npy' or ext == 'htk', '{} is an unknown extention'.format(ext)

mean_buf = None
var_buf = None

v_true = None
y_true = None

num_frame = 0;
used_frame = 0

f = open(script_filename)
for line in f:
    s = line.strip()
    #x_file, y_file = s.split(' ', 1)
    x_file = s.strip()
    y_flle = x_file
    print(x_file)
    if not os.path.exists(x_file):
        continue  
    assert '.'+ext in x_file, 'I am not sure that {} is a {} file. Please check it.'.format(x_file, ext)
    if '.htk' in x_file:
        fh = open(x_file, "rb")
        spam = fh.read(12)
        nSamples, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)
        veclen = sampSize / 4
        fh.seek(12, 0)
        utt_data = np.fromfile(fh, 'f')
        #print(veclen, len(utt_data)/veclen)
        utt_data = utt_data.reshape(int(len(utt_data)/veclen), int(veclen))
        utt_data = utt_data.T
        utt_data = utt_data.byteswap()
        fh.close()
    elif '.npy' in x_file:
        utt_data = np.load(x_file).T

    if mean_buf is None:
        mean_buf = np.zeros((utt_data.shape[0], 1))
    if var_buf is None:
        var_buf = np.zeros((utt_data.shape[0], 1))

    num_frame += utt_data.shape[1]

    mean_buf += np.sum(utt_data, axis=1)[:, np.newaxis]
    var_buf += np.sum(utt_data**2, axis=1)[:, np.newaxis]

f.close()

mean_buf /= num_frame
var_buf /= num_frame
var_buf += - mean_buf ** 2

#if calc_normalize is False:
#    np.save('mean.npy', mean_buf)
#    np.save('var.npy', var_buf)
#    print("mean and var save end.")
#    sys.exit(1)
#else:
#    print('Normalize')

f = open(script_filename)
for line in f:
    s = line.strip()
    #x_file, y_file = s.split(' ', 1)
    x_file = s.strip()
    y_file = x_file
                        
    if '.htk' in x_file:
        if not os.path.exists(x_file):
            continue
        fh = open(x_file, "rb")
        spam = fh.read(12)
        nSamples, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)
        veclen = sampSize / 4
        fh.seek(12, 0)
        utt_data = np.fromfile(fh, 'f')
        utt_data = utt_data.reshape(int(len(utt_data)/veclen), int(veclen))
        utt_data = utt_data.T
        utt_data = utt_data.byteswap()
        fh.close()
    if '.npy' in x_file:
        utt_data = np.load(x_file).T

    utt_data -= mean_buf
    #utt_data /= var_buf ** (0.5)
    utt_data /= np.sqrt(var_buf)

    if '.htk' in x_file:
        fh = open(y_file, "wb")
        fh.seek(0,0)
        fh.write(pack(">IIHH", nSamples, sampPeriod, sampSize, parmKind))
        for raw in utt_data.T:
            np.array(raw, 'f').byteswap().tofile(fh)
    elif '.npy' in x_file:
        np.save(x_file, utt_data.T)

f.close()
