# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np
import librosa
import collections
import os
import random
from struct import unpack
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import sentencepiece as spm

class TrainDatasets(Dataset):
    """
    Train dataset for ASR.
    """                                                   
    def __init__(self, csv_file, hp, use_spec_aug=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations
            use_spec_aug (bool): Use SpecAugment or not
        """
        self.landmarks_frame = pd.read_csv(csv_file, sep='\|', header=None)
        self.use_spec_aug = use_spec_aug
        self.hp = hp
        if hp.spm_model is not None:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(hp.spm_model)

        if hp.mean_file is not None and hp.var_file is not None:
            self.mean_value = np.load(hp.mean_file).reshape(-1, hp.lmfb_dim)
            self.var_value = np.load(hp.var_file).reshape(-1, hp.lmfb_dim)
                                                                                
    def load_wav(self, filename):
        return librosa.load(filename, sr=self.hp.sample_rate)

    def load_htk(self, filename):
        with open(filename, "rb") as fh:
            spam = fh.read(12)
            _, _, sampSize, _ = unpack(">IIHH", spam)
            veclen = int(sampSize / 4)
            fh.seek(12, 0)
            dat = np.fromfile(fh, dtype='float32')
            dat = dat.reshape(int(len(dat) / veclen), veclen)
            dat = dat.byteswap()
        return dat

    def __len__(self):
        return len(self.landmarks_frame)

    def _freq_mask(self, spec, F=10, num_masks=1, replace_with_zero=False):
        cloned = spec.clone()
        num_mel_channels = cloned.shape[1]

        for _ in range(0, num_masks):
            f = random.randrange(0, F)
            f_zero = random.randrange(0, num_mel_channels - f)

            # avoids randrange error if values are equal and range is empty
            if (f_zero == f_zero + f): return cloned

            mask_end = random.randrange(f_zero, f_zero + f)
            if (replace_with_zero):
                cloned[:, f_zero:mask_end] = 0
            else:
                cloned[:, f_zero:mask_end] = cloned.mean()

        return cloned

    def _time_mask(self, spec, T=50, num_masks=1, replace_with_zero=False):
        cloned = spec.clone()
        len_spectro = cloned.shape[0]

        for i in range(0, num_masks):
            t = random.randrange(0, T)
            t_zero = random.randrange(0, len_spectro - t)

            # avoids randrange error if values are equal and range is empty
            if (t_zero == t_zero + t): return cloned

            mask_end = random.randrange(t_zero, t_zero + t)
            if (replace_with_zero):
                cloned[t_zero:mask_end, :] = 0
            else:
                cloned[t_zero:mask_end, :] = cloned.mean()
        return cloned

    def __getitem__(self, idx):
        mel_name = self.landmarks_frame.loc[idx, 0]
        text = self.landmarks_frame.loc[idx, 1].strip()

        if self.hp.spm_model is not None:
            textids = [self.sp.bos_id()] + self.sp.EncodeAsIds(text)+ [self.sp.eos_id()]
            text = np.array([int(t) for t in textids], dtype=np.int32)
        else:
            text = np.array([int(t) for t in text.split(' ')], dtype=np.int32)
        if '.npy' in mel_name:
            mel_input = np.load(mel_name)
            assert mel_input.shape[0] == self.hp.lmfb_dim or mel_input.shape[1] == self.hp.lmfb_dim, '{} does not have strange shape {}'.format(mel_name, mel_input.shape)
            if mel_input.shape[1] != self.hp.lmfb_dim:
                mel_input = mel_input.T
        elif '.htk' in mel_name:
            mel_input = self.load_htk(mel_name)[:,:self.hp.lmfb_dim]
        else:
            raise ValueError('{} is unknown file extension. Please check the extension or change htk or npy'.format(mel_name))

        if self.hp.mean_file is not None and self.hp.var_file is not None:
            mel_input -= self.mean_value
            mel_input /= np.sqrt(self.var_value)

        if self.use_spec_aug:
            mel_input = torch.from_numpy(mel_input)
            mel_length = mel_input.shape[0]
            T = min(mel_input.shape[0] // 2 - 1, self.hp.max_width_T)
            #num_T = min(20, math.floor(0.04*mel_length))
            #T = math.floor(0.04*mel_length)
            mel_input = self._time_mask(self._freq_mask(mel_input, F=self.hp.max_width_F, num_masks=self.hp.num_mask_F), T=T, num_masks=self.hp.num_mask_T, replace_with_zero=True)
            #mel_input = self._time_mask(self._freq_mask(mel_input, F=hp.spec_size_f, num_masks=2, replace_with_zero=True), T=T, num_masks=num_T, replace_with_zero=True)
        text_length = len(text)
        mel_length = mel_input.shape[0]
        pos_text = np.arange(1, text_length + 1)
        pos_mel = np.arange(1, mel_length + 1)

        sample = {'text': text, 'text_length': text_length, 'mel_input': mel_input, 'mel_length': mel_length, 'pos_mel': pos_mel, 'pos_text': pos_text}

        return sample


class TestDatasets(Dataset):
    """
    Test dataset.
    """
    def __init__(self, csv_file, hp, align=False, tts_filter=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations
            root_dir (string): Directory with all the wavs
        """
        # self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.landmarks_frame = pd.read_csv(csv_file, sep='\|', header=None)
        self.align = align
        self.hp = hp
        self.tts_filter = tts_filter
        if self.hp.spm_model is not None:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(hp.spm_model)

        if self.hp.mean_file is not None and self.hp.var_file is not None:
            self.mean_value = np.load(self.hp.mean_file).reshape(-1, self.hp.lmfb_dim)
            self.var_value = np.load(self.hp.var_file).reshape(-1, self.hp.lmfb_dim)

    def load_wav(self, filename):
        return librosa.load(filename, sr=self.hp.sample_rate)

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

        if '.npy' in mel_name:
            mel_input = np.load(mel_name)
            assert mel_input.shape[0] == self.hp.lmfb_dim or mel_input.shape[1] == self.hp.lmfb_dim, '{} does not have strange shape {}'.format(mel_name, mel_input.shape)
            if mel_input.shape[1] != self.hp.lmfb_dim:
                mel_input = mel_input.T
        elif '.htk' in mel_name:
            mel_input = self.load_htk(mel_name)[:,:self.hp.lmfb_dim]
        else:
            raise ValueError('{} is unknown file extension. Please check the extension or change htk or npy'.format(mel_name))

        if self.hp.mean_file is not None and self.hp.var_file is not None:
            mel_input -= self.mean_value
            mel_input /= np.sqrt(self.var_value)

        mel_length = mel_input.shape[0]
        pos_mel = np.arange(1, mel_length + 1)

        if self.align:
            text = self.landmarks_frame.loc[idx, 1].strip()
            if self.hp.spm_model is not None:
                textids = [self.sp.bos_id()] + self.sp.EncodeAsIds(text)+ [self.sp.eos_id()]
                text = np.array([int(t) for t in textids], dtype=np.int32)
            else:
                text = np.array([int(t) for t in text.split(' ')], dtype=np.int32)

            text_length = len(text)
            pos_text = np.arange(1, text_length + 1)

            sample = {'text': text, 'text_length': text_length, 'mel_input': mel_input, 'mel_length': mel_length, 'pos_mel': pos_mel, 'pos_text': pos_text, 'mel_name': mel_name}

        elif self.tts_filter:
            text = self.landmarks_frame.loc[idx, 1].strip()
            text_raw = self.landmarks_frame.loc[idx, 2].strip()
            if self.hp.spm_model is not None:
                textids = [self.sp.bos_id()] + self.sp.EncodeAsIds(text) + [self.sp.eos_id()]
                text = np.array([int(t) for t in textids], dtype=np.int32)
            else:
                text = np.array([int(t) for t in text.split(' ')], dtype=np.int32)

            text_length = len(text)
            pos_text = np.arange(1, text_length + 1)

            sample = {'text': text, 'text_length': text_length, 'mel_input': mel_input, 'mel_length': mel_length, 'pos_mel': pos_mel, 'pos_text': pos_text, 'mel_name': mel_name, 'text_raw': text_raw}
        else:
            sample = {'mel_input': mel_input, 'mel_length': mel_length, 'pos_mel': pos_mel, 'mel_name': mel_name}

        return sample


def collate_fn(batch):
    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.abc.Mapping):
        text = [d['text'] for d in batch]
        mel_input = [d['mel_input'] for d in batch]
        text_length = [d['text_length'] for d in batch]
        mel_length = [d['mel_length'] for d in batch]
        pos_mel = [d['pos_mel'] for d in batch]
        pos_text = [d['pos_text'] for d in batch]

        text = [i for i, _ in sorted(zip(text, mel_length), key=lambda x: x[1], reverse=True)]
        mel_input = [i for i, _ in sorted(zip(mel_input, mel_length), key=lambda x: x[1], reverse=True)]
        pos_text = [i for i, _ in sorted(zip(pos_text, mel_length), key=lambda x: x[1], reverse=True)]
        pos_mel = [i for i, _ in sorted(zip(pos_mel, mel_length), key=lambda x: x[1], reverse=True)]
        text_length = [i for i, _ in sorted(zip(text_length, mel_length), key=lambda x: x[1], reverse=True)]
        mel_length = sorted(mel_length, reverse=True)
        # PAD sequences with largest length of the batch
        text = _prepare_data(text).astype(np.int32)
        mel_input = _pad_mel(mel_input)
        pos_mel = _prepare_data(pos_mel).astype(np.int32)
        pos_text = _prepare_data(pos_text).astype(np.int32)

        return torch.LongTensor(text), torch.FloatTensor(mel_input), torch.LongTensor(pos_text), torch.LongTensor(pos_mel), torch.LongTensor(text_length), torch.LongTensor(mel_length)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}".format(type(batch[0]))))


def collate_fn_align(batch):
    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.abc.Mapping):
        text = [d['text'] for d in batch]
        mel_input = [d['mel_input'] for d in batch]
        text_length = [d['text_length'] for d in batch]
        mel_length = [d['mel_length'] for d in batch]
        pos_mel = [d['pos_mel'] for d in batch]
        pos_text = [d['pos_text'] for d in batch]
        mel_name = [d['mel_name'] for d in batch]

        # PAD sequences with largest length of the batch
        text = _prepare_data(text).astype(np.int32)
        mel_input = _pad_mel(mel_input)
        pos_mel = _prepare_data(pos_mel).astype(np.int32)
        pos_text = _prepare_data(pos_text).astype(np.int32)

        return torch.LongTensor(text), torch.FloatTensor(mel_input), torch.LongTensor(pos_text), torch.LongTensor(pos_mel), torch.LongTensor(text_length), torch.LongTensor(mel_length), mel_name

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}".format(type(batch[0]))))


def collate_fn_tts_filter(batch):
    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.abc.Mapping):
        text = [d['text'] for d in batch]
        mel_input = [d['mel_input'] for d in batch]
        text_length = [d['text_length'] for d in batch]
        mel_length = [d['mel_length'] for d in batch]
        pos_mel = [d['pos_mel'] for d in batch]
        pos_text = [d['pos_text'] for d in batch]
        mel_name = [d['mel_name'] for d in batch]
        text_raw = [d['text_raw'] for d in batch]

        # PAD sequences with largest length of the batch
        text = _prepare_data(text).astype(np.int32)
        mel_input = _pad_mel(mel_input)
        pos_mel = _prepare_data(pos_mel).astype(np.int32)
        pos_text = _prepare_data(pos_text).astype(np.int32)

        return torch.LongTensor(text), torch.FloatTensor(mel_input), torch.LongTensor(pos_text), torch.LongTensor(pos_mel), torch.LongTensor(text_length), torch.LongTensor(mel_length), mel_name, text_raw

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}".format(type(batch[0]))))


def collate_fn_test(batch):
    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.abc.Mapping):
        mel_input = [d['mel_input'] for d in batch]
        mel_length = [d['mel_length'] for d in batch]
        pos_mel = [d['pos_mel'] for d in batch]

        mel_input = _pad_mel(mel_input)
        pos_mel = _prepare_data(pos_mel).astype(np.int32)

        return torch.FloatTensor(mel_input), torch.LongTensor(pos_mel), torch.LongTensor(mel_length)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}".format(type(batch[0]))))


def _pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])


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
        return np.pad(x, [[0, max_len - mel_len], [0, 0]], mode='constant', constant_values=_pad)
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
    """
    Sampler for extracting batch data whose size is fixed size
    """
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
        all_indices = np.arange(self.dataset_len - mod).reshape(batch_len, self.batch_size).tolist()
        if mod != 0:
            remained = np.arange(self.dataset_len - mod, self.dataset_len).tolist()
            all_indices.append(remained)

        return all_indices

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.all_indices)

        for indices in self.all_indices:
            yield indices

    def __len__(self):
        return len(self.all_indices)


def get_dataset(script_file, hp, use_spec_aug):
    print(f'script_file = {script_file}')
    print(f'use_spec_aug = {use_spec_aug}')
    return TrainDatasets(script_file, hp, use_spec_aug=use_spec_aug)


if __name__ == '__main__':
    from utils import hparams as hp
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py')
    parser.add_argument('--train_script', default=None)
    args = parser.parse_args()

    hp.configure(args.hp_file)
    if args.train_script is not None:
        hp.train_script = args.train_script
    print(f'train script = {hp.train_script}')
    datasets = TrainDatasets(hp.train_script, hp)
    sampler = NumBatchSampler(datasets, hp.batch_size, True, True)
    dataloader = DataLoader(datasets, batch_sampler=sampler, num_workers=4, collate_fn=collate_fn)
    from tqdm import tqdm
    pbar = tqdm(dataloader)
    for d in pbar:
        pass
        #print(d[1].shape)
