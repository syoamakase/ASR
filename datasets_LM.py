# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import librosa
import collections
import os
import random
from struct import unpack, pack
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import librosa
import sentencepiece as spm

from utils import hparams as hp

class TrainDatasets(Dataset):                                                     
    """
    Train dataset.
    """                                                   
    def __init__(self, csv_file):                                     
        """                                                                     
        Args:                                                                   
            csv_file (string): Path to the csv file with annotations.           
            root_dir (string): Directory with all the wavs.                     
                                                                                
        """                                                                     
        # self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)  
        self.landmarks_frame = pd.read_csv(csv_file, sep='\|', header=None)    
        if hp.spm_model is not None:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(hp.spm_model)

    def __len__(self):
        return len(self.landmarks_frame) 
                                        
    def __getitem__(self, idx): 
        # mel_name = self.landmarks_frame.loc[idx, 0]
        text = str(self.landmarks_frame.loc[idx, 0]).strip()
                                                                                
        if hp.spm_model is not None:
            textids = [self.sp.bos_id()] + self.sp.EncodeAsIds(text)+ [self.sp.eos_id()]
            text = np.array([int(t) for t in textids], dtype=np.int32)
        else:
            text = np.array([int(t) for t in text.split(' ')], dtype=np.int32)

        text_length = len(text)                             
        pos_text = np.arange(1, text_length + 1)
                                                                                
        sample = {'text': text, 'text_length':text_length, 'pos_text':pos_text}
                                                                                
        return sample

class TestDatasets(Dataset):                                                     
    """
    Train dataset.
    """                                                   
    def __init__(self, csv_file):                                     
        """                                                                     
        Args:                                                                   
            csv_file (string): Path to the csv file with annotations.           
            root_dir (string): Directory with all the wavs.                     
                                                                                
        """                                                                     
        # self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)  
        self.landmarks_frame = pd.read_csv(csv_file, sep='\|', header=None)    
        if hp.spm_model is not None:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(hp.spm_model)

    def __len__(self):
        return len(self.landmarks_frame) 
                                        
    def __getitem__(self, idx): 
        mel_name = self.landmarks_frame.loc[idx, 0]
        raw_text = str(self.landmarks_frame.loc[idx, 1]).strip()
                                                                                
        if hp.spm_model is not None:
            textids = [self.sp.bos_id()] + self.sp.EncodeAsIds(raw_text) + [self.sp.eos_id()]
            text = np.array([int(t) for t in textids], dtype=np.int32)
        else:
            text = np.array([int(t) for t in text.split(' ')], dtype=np.int32)

        text_length = len(text)                             
        pos_text = np.arange(1, text_length + 1)
                                                                                
        sample = {'text': text, 'text_length':text_length, 'pos_text':pos_text, 'mel_name': mel_name, 'raw_text':raw_text}         
        return sample

def collate_fn(batch):
    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.abc.Mapping):
        text = [d['text'] for d in batch]
        text_length = [d['text_length'] for d in batch]
        pos_text = [d['pos_text'] for d in batch]
        
        text = [i for i,_ in sorted(zip(text, text_length), key=lambda x: x[1], reverse=True)]
        pos_text = [i for i, _ in sorted(zip(pos_text, text_length), key=lambda x: x[1], reverse=True)]
        text_length = sorted(text_length, reverse=True)
        text = _prepare_data(text).astype(np.int32)
        pos_text = _prepare_data(pos_text).astype(np.int32)

        return torch.LongTensor(text), torch.LongTensor(pos_text), torch.LongTensor(text_length)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

def collate_fn_test(batch):
    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.abc.Mapping):
        text = [d['text'] for d in batch]
        text_length = [d['text_length'] for d in batch]
        pos_text = [d['pos_text'] for d in batch]
        mel_name = [d['mel_name'] for d in batch]
        raw_text = [d['raw_text'] for d in batch]
        
        text = [i for i,_ in sorted(zip(text, text_length), key=lambda x: x[1], reverse=True)]
        mel_name = [i for i,_ in sorted(zip(mel_name, text_length), key=lambda x: x[1], reverse=True)]
        pos_text = [i for i, _ in sorted(zip(pos_text, text_length), key=lambda x: x[1], reverse=True)]
        raw_text = [i for i, _ in sorted(zip(raw_text, text_length), key=lambda x: x[1], reverse=True)]
        text_length = sorted(text_length, reverse=True)
        text = _prepare_data(text).astype(np.int32)
        pos_text = _prepare_data(pos_text).astype(np.int32)

        return torch.LongTensor(text), torch.LongTensor(pos_text), torch.LongTensor(text_length), mel_name, raw_text

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

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

def _pad_mel(inputs):
    _pad = 0
    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(x, [[0,max_len - mel_len],[0,0]], mode='constant', constant_values=_pad)
    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])

class LengthsBatchSampler(Sampler):
    """
    LengthsBatchSampler - Sampler for variable batch size. Mainly, we use it for Transformer.
    It requires lengths.
    """
    def __init__(self, dataset, n_lengths, lengths_file=None, shuffle=True, shuffle_one_time=False, reverse=False):
        assert not ((shuffle == reverse) and shuffle is True), 'shuffle and reverse cannot set True at the same time.'

        if lengths_file is None or not os.path.exists(lengths_file):
            print('lengths_file is not exists. Make...')
            loader = DataLoader(dataset, num_workers=1)
            lengths_list = []
            pbar = tqdm(loader)
            for d in pbar:
                mel_input = d['mel_input']
                lengths_list.append(mel_input.shape[1])
            self.lengths_np = np.array(lengths_list)
            np.save('lengths.npy', self.lengths_np)
        else:
            print('{} is loading.'.format(lengths_file))
            self.lengths_np = np.load(lengths_file)
            assert len(dataset) == len(self.lengths_np), 'mismatch the number of lines between dataset and {}'.format(lengths_file)
        
        self.n_lengths = n_lengths
        self.all_indices = self._batch_indices()
        if shuffle_one_time:
            random.shuffle(self.all_indices)
        self.shuffle = shuffle
        self.shuffle_one_time = shuffle_one_time
        self.reverse = reverse

    def _batch_indices(self):
        self.count = 0
        all_indices = []
        while self.count + 1 < len(self.lengths_np):
            indices = []
            mel_lengths = 0
            while self.count < len(self.lengths_np):
                curr_len = self.lengths_np[self.count]
                if mel_lengths + curr_len > self.n_lengths or (self.count + 1) > len(self.lengths_np):
                    break
                mel_lengths += curr_len
                indices.extend([self.count])
                self.count += 1
            all_indices.append(indices)
       
        return all_indices

    def __iter__(self):
        if self.shuffle and not self.shuffle_one_time:
            random.shuffle(self.all_indices)
        if self.reverse:
            self.all_indices.reverse()

        for indices in self.all_indices:
            yield indices

    def __len__(self):
        return len(self.all_indices)

class NumBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=True, shuffle=True):
        self.batch_size = batch_size
        self.drop_last = drop_last 
        self.dataset_len = len(dataset)
        self.all_indices = self._batch_indices()
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.all_indices)

    def _batch_indices(self):
        batch_len = self.dataset_len // self.batch_size
        mod = self.dataset_len % self.batch_size
        all_indices = np.arange(self.dataset_len-mod).reshape(batch_len, self.batch_size).tolist()
        if mod != 0:
            remained = np.arange(self.dataset_len-mod, self.dataset_len).tolist()
            all_indices.append(remained) 
       
        return all_indices

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.all_indices)

        for indices in self.all_indices:
            yield indices

    def __len__(self):
        return len(self.all_indices)

def get_dataset(script_file):
    print(f'script_file = {script_file}')
    return TrainDatasets(script_file)

if __name__ == '__main__':
    datasets = get_dataset()
    sampler = LengthsBatchSampler(datasets, 58000, 'lengths.npy', True, True, False)
    dataloader = DataLoader(datasets, batch_sampler=sampler, num_workers=4, collate_fn=collate_fn_transformer)
    from tqdm import tqdm
    pbar = tqdm(dataloader)
    for d in pbar:
        print(d[1].shape)
