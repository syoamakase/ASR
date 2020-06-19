#-*- coding: utf-8 -*-
import argparse
import librosa
import numpy as np
import os
import random
from struct import unpack, pack
import sys
import torch
import warnings

def load_dat(filename):
    fh = open(filename, "rb")
    spam = fh.read(12)
    nSamples, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)
    veclen = int(sampSize / 4)
    fh.seek(12, 0)
    dat = np.fromfile(fh, dtype=np.float32)
    dat = dat.reshape(int(len(dat) / veclen), veclen)
    dat = dat.byteswap()
    fh.close()
    return dat

def sort_by_xlen(script_filename, mel_dim, threshold=100000):
    omitted_label = []

    with open(script_filename, 'r') as f:
        a = []    
        for line in f:
            sp_line = line.strip().split('|')
            binname = sp_line[0]
            if not os.path.exists(binname):
                warnings.warn('{} not found'.format(binname))
                continue
            if '.htk' in binname:
                x = load_dat(binname)
                if threshold > len(x):
                    a.append([line, len(x)])
                else:
                    omitted_label.append([line, len(x)])
            elif '.npy' in binname:
                x = numpy.load(binname)
                if threshold > len(x):
                    a.append([line, len(x)])
                else:
                    omitted_label.append([line, len(x)])
            elif '.pt' in binname:
                x = torch.load(binname)
                if len(x) == mel_dim:
                    x = x.transpose(0,1)
                if threshold > len(x): 
                    a.append([line, len(x)])
                else:
                    omitted_label.append([line, len(x)])
            elif '.wav' in binname:
                x, sr =  librosa.load(binname)
                a.append([line, x.shape[0]])
            else:
                print('Error')
                sys.exit(1)

    random.shuffle(a)
    a_sort = sorted(a, key=lambda x:x[1])

    for k in a_sort:
        print(k[0].strip()+'|'+str(k[1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--script_filename', required=True)
    parser.add_argument('-m', '--mel_dim', type=int, default=80)
    args = parser.parse_args()
    script_filename = args.script_filename
    mel_dim = args.mel_dim

    sort_by_xlen(script_filename=script_filename, mel_dim=mel_dim)


