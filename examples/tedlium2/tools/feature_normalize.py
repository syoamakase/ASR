#-*- coding:utf-8 -*-
import argparse
import os
import numpy as np
from struct import unpack

def calc_mean_var(script_filename,save_dir, ext):
    mean_buf = None
    var_buf = None
    num_frame = 0
    with open(script_filename) as f:
        for line in f:
            x_file = line.strip()
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

    mean_buf /= num_frame
    var_buf /= num_frame
    var_buf += - mean_buf ** 2

    np.save(os.path.join(save_dir, 'mean.npy'), mean_buf)
    np.save(os.path.join(save_dir, 'var.npy'), var_buf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--script_filename', required=True)
    parser.add_argument('--ext', required=True)
    parser.add_argument('--save_dir', required=True)
    args = parser.parse_args()
    script_filename = args.script_filename
    save_dir = args.save_dir
    ext = args.ext

    assert ext == 'npy' or ext == 'htk', '{} is an unknown extention'.format(ext)

    calc_mean_var(script_filename, save_dir, ext)

