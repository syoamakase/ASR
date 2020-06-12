
import argparse
import os
import sys
import time
import numpy as np
import random
from struct import unpack, pack


np.seterr(all='warn')

parser = argparse.ArgumentParser()
parser.add_argument('-S', '--script_filename', required=True)
parser.add_argument('--htk_format', action='store_true')
args = parser.parse_args()
script_filename = args.script_filename
htk_format = args.htk_format

mean_buf = None
var_buf = None

v_true = None
y_true = None

num_frame = 0
used_frame = 0

with open(script_filename) as f:
    for line in f:
        if len(line.split(' ')) != 1:
            x_file, _ = line.strip().split(' ', 1)
        else:
            x_file = line.strip()
        #print(x_file)
        if not os.path.exists(x_file):
            continue            
        if htk_format:
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
        else:
            utt_data = np.load(x_file).T
    
        if mean_buf is None:
            mean_buf = np.zeros((utt_data.shape[0], 1))
        if var_buf is None:
            var_buf = np.zeros((utt_data.shape[0], 1))
    
        num_frame += utt_data.shape[1]
    
        mean_buf += np.sum(utt_data, axis=1)[:, np.newaxis]
        var_buf += np.sum(utt_data**2, axis=1)[:, np.newaxis]

mean_buf /= num_frame
var_buf /= num_frame
var_buf += - mean_buf ** 2

with open(script_filename) as f:
    for line in f:
        if len(line.split(' ')) != 1:
            x_file, _ = line.strip().split(' ', 1)
        else:
            x_file = line.strip()
        if not os.path.exists(x_file):
            continue

        if htk_format:
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
            utt_data -= mean_buf
            utt_data /= np.sqrt(var_buf)
        
            fh = open(x_file, "wb")
            fh.seek(0,0)
            for raw in utt_data.T:
                np.array(raw, 'f').byteswap().tofile(fh)
                
        else:
            utt_data = np.load(x_file).T
            utt_data -= mean_buf
            utt_data /= np.sqrt(var_buf)

            np.save(x_file, utt_data.T)
