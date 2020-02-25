# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import librosa
import collections
import os
from struct import unpack, pack
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
from sphfile import SPHFile

class TedliumPreprocess(Dataset):                                                     
    """
    A preprocess for Tedlium datasets

    """                                                   
    def __init__(self, wav_scp_file, segments_file, save_dir=None):                                     
        """                                                                     
        Args:                                                                   
            csv_file (string): Path to the csv file with annotations.           
            root_dir (string): Directory with all the wavs.                     
                                                                                
        """                                                                     
        # self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)  
        # self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.wav_scp_landmarks = pd.read_csv(wav_scp_file, sep=' ', header=None)
        self.segments_landmarks = pd.read_csv(segments_file, sep=' ', header=None)
        self.sph_file_dict = self._extract_file_name()

        self.save_dir_wav = os.path.join(save_dir, 'wav')
        self.save_dir_lmfb = os.path.join(save_dir, 'lmfb')

    def load_wav(self, filename):                                               
        return librosa.load(filename, sr=hp.sample_rate) 

    def load_htk(self, filename):
        fh = open(filename, "rb")
        spam = fh.read(12)
        _, _, sampSize, _ = unpack(">IIHH", spam)
        veclen = int(sampSize / 4)
        fh.seek(12, 0)
        dat = np.fromfile(fh, dtype='float32')
        dat = dat.reshape(int(len(dat) / veclen), veclen)
        dat = dat.byteswap()
        fh.close()
        return dat

    # def calc_lmfb(self):

    def _extract_file_name(self):
        results_dict = {}
        for i in range(self.wav_scp_landmarks.shape[0]):
            sph_name = self.wav_scp_landmarks.loc[i, 5]
            file_id = os.path.splitext(os.path.basename(sph_name))[0]
            results_dict[file_id] = sph_name
        return results_dict
                                                                                
    def __len__(self):                                                          
        return len(self.segments_landmarks)                                        
                                                                                
    def __getitem__(self, idx): 
        save_id, file_id, start_time, end_time = self.segments_landmarks.loc[idx, :]

        # text = np.array([int(t) for t in text.split(' ')], dtype=np.int32)
        sph_name = self.sph_file_dict[file_id]
        sph = SPHFile(sph_name)
        dir_name = os.path.join(self.save_dir_wav, file_id)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        save_name = os.path.join(dir_name, save_id)
        sph.write_wav(save_name+'.wav', float(start_time), float(end_time))

        sample = {'dummy':0}
        return sample

def _pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])

def _pad_per_step(inputs):
    timesteps = inputs.shape[-1]
    return np.pad(inputs, [[0,0],[0,0],[0, hp.outputs_per_step - (timesteps % hp.outputs_per_step)]], mode='constant', constant_values=0.0)

def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params

def get_dataset():
    return TedliumPreprocess('/n/work1/ueno/data/tedlium3/data/train/wav.scp', '/n/work1/ueno/data/tedlium3/data/train/segments', '/n/work1/ueno/data/tedlium3') 

def _pad_mel(inputs):
    _pad = 0
    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(x, [[0,max_len - mel_len],[0,0]], mode='constant', constant_values=_pad)
    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])

if __name__ == '__main__':
    datasets = get_dataset()
    # num_workers must be set 1 !!
    dataloader = DataLoader(datasets, batch_size=1, drop_last=False, num_workers=1)
    from tqdm import tqdm
    pbar = tqdm(dataloader)
    for d in pbar:
        pass
