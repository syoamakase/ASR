
from struct import unpack, pack
import numpy
import os
import random
import sys
import torch
import warnings

def load_dat(filename):
    fh = open(filename, "rb")
    spam = fh.read(12)
    nSamples, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)
    veclen = int(sampSize / 4)
    fh.seek(12, 0)
    dat = numpy.fromfile(fh, dtype=numpy.float32)
    dat = dat.reshape(int(len(dat) / veclen), veclen)
    dat = dat.byteswap()
    fh.close()
    return dat

filename = sys.argv[1]
threshold = 150000
mel_dim = 40

omitted_label = []

with open(filename, 'r') as f:
    a = []    
    for line in f:
        sp_line = line.split(' ')
        binname = sp_line[0]
        if not os.path.exists(binname):
            warnings.warn('{} not found'.format(binname))
            continue
        if '.htk' in binname:
            x = load_dat(binname)
            #print(sp_line, len(x))
            if threshold > len(x):
                a.append([line, len(x)])
            else:
                omitted_label.append([line, len(x)])
        elif '.npy' in binname:
            x = numpy.load(binname)
            #print(sp_line, len(x))
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

        else:
            print('Error')
            sys.exit(1)

random.shuffle(a)
a_sort = sorted(a, key=lambda x:x[1])

for k in a_sort:
    print(k[0].strip()+'|'+str(k[1]))

if len(omitted_label) != 0:
    print(omitted_label)
