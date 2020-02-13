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

class TrainDatasets(Dataset):                                                     
    """
    Train dataset.

    """                                                   
    def __init__(self, csv_file, root_dir=None):                                     
        """                                                                     
        Args:                                                                   
            csv_file (string): Path to the csv file with annotations.           
            root_dir (string): Directory with all the wavs.                     
                                                                                
        """                                                                     
        # self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)  
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir                                                
                                                                                
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
                                                                                
    def __len__(self):                                                          
        return len(self.landmarks_frame)                                        
                                                                                
    def __getitem__(self, idx): 
        mel_name = self.landmarks_frame.loc[idx, 0]
        text = self.landmarks_frame.loc[idx, 1].strip()
                                                                                
        # text = np.asarray(text_to_sequence(text, [hp.cleaners]), dtype=np.int32)
        text = np.array([int(t) for t in text.split(' ')], dtype=np.int32)
        if '.npy' in mel_name:
            mel_input = np.load(mel_name)
        elif '.htk' in mel_name:
            mel_input = self.load_htk(mel_name)[:,:40]
        # mel_input = np.concatenate([np.zeros([1,hp.num_mels], np.float32), mel[:-1,:]], axis=0)
        text_length = len(text)
        mel_length = mel_input.shape[0]                              
        pos_text = np.arange(1, text_length + 1)
        pos_mel = np.arange(1, mel_length + 1) 
                                                                                
        sample = {'text': text, 'text_length':text_length, 'mel_input':mel_input, 'mel_length':mel_length, 'pos_mel':pos_mel, 'pos_text':pos_text}
                                                                                
        return sample

def collate_fn(batch):
    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.abc.Mapping):
        text = [d['text'] for d in batch]
        mel_input = [d['mel_input'] for d in batch]
        text_length = [d['text_length'] for d in batch]
        mel_length = [d['mel_length'] for d in batch]
        pos_mel = [d['pos_mel'] for d in batch]
        pos_text= [d['pos_text'] for d in batch]
        
        #text = [i for i,_ in sorted(zip(text, text_length), key=lambda x: x[1], reverse=True)]
        #mel_input = [i for i, _ in sorted(zip(mel_input, text_length), key=lambda x: x[1], reverse=True)]
        #pos_text = [i for i, _ in sorted(zip(pos_text, text_length), key=lambda x: x[1], reverse=True)]
        #pos_mel = [i for i, _ in sorted(zip(pos_mel, text_length), key=lambda x: x[1], reverse=True)]
        #text_length = sorted(text_length, reverse=True)
        # PAD sequences with largest length of the batch
        text = _prepare_data(text).astype(np.int32)
        mel_input = _pad_mel(mel_input)
        pos_mel = _prepare_data(pos_mel).astype(np.int32)
        pos_text = _prepare_data(pos_text).astype(np.int32)

        return torch.LongTensor(text), torch.FloatTensor(mel_input), torch.LongTensor(pos_text), torch.LongTensor(pos_mel), torch.LongTensor(text_length), torch.LongTensor(mel_length)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

def collate_fn_transformer(batch):
    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.abc.Mapping):
        text = [d['text'] for d in batch]
        mel_input = [d['mel_input'] for d in batch]
        text_length = [d['text_length'] for d in batch]
        mel_length = [d['mel_length'] for d in batch]
        pos_mel = [d['pos_mel'] for d in batch]
        pos_text= [d['pos_text'] for d in batch]
        
        #text = [i for i,_ in sorted(zip(text, text_length), key=lambda x: x[1], reverse=True)]
        #mel_input = [i for i, _ in sorted(zip(mel_input, text_length), key=lambda x: x[1], reverse=True)]
        #pos_text = [i for i, _ in sorted(zip(pos_text, text_length), key=lambda x: x[1], reverse=True)]
        #pos_mel = [i for i, _ in sorted(zip(pos_mel, text_length), key=lambda x: x[1], reverse=True)]
        #text_length = sorted(text_length, reverse=True)
        # PAD sequences with largest length of the batch
        text = _prepare_data(text).astype(np.int32)
        mel_input = _pad_mel(mel_input)
        pos_mel = _prepare_data(pos_mel).astype(np.int32)
        pos_text = _prepare_data(pos_text).astype(np.int32)

        return torch.LongTensor(text), torch.FloatTensor(mel_input), torch.LongTensor(pos_text), torch.LongTensor(pos_mel), torch.LongTensor(text_length), torch.LongTensor(mel_length)

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

def get_dataset(script_file='/home/ueno/Datasets/csj/script.bpe.sort_xlen'):
    print('script_file = {}'.format(script_file))
    return TrainDatasets(script_file)

if __name__ == '__main__':
    datasets = get_dataset()
    sampler = LengthsBatchSampler(datasets, 58000, '/home/ueno/Datasets/csj/lengths.npy', True, True, False)
    dataloader = DataLoader(datasets, batch_sampler=sampler, num_workers=4, collate_fn=collate_fn_transformer)
    from tqdm import tqdm
    pbar = tqdm(dataloader)
    for d in pbar:
        print(d[1].shape)
