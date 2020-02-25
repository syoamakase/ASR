# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import librosa
import collections
import os
from struct import unpack, pack
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io.wavfile
from sphfile import SPHFile

# メルスケールの定義(m = 1127 * ln(f / 700Hz) + 1))
def Mel(k, fresh):
    return 1127 * np.log(1 + (k - 1) * fres)
# 高域強調の定数
pre_emphasis = 0.97
# フレームの窓幅(400サンプル=25ms)
frame_length = 400
# フレームのずらす幅(160サンプル=10ms)
frame_step = 160
# FFTのサイズ
NFFT = 512
# ナイキスト周波数に対応するビン
Nby2 = int(NFFT / 2)
# メルスケールの計算に必要な係数
fres = 16000 / (NFFT * 700)
# フィルタバンクの計算に使う最小周波数
klo = 2
# フィルタバンクの計算に使う最大周波数
khi = Nby2
# フィルタバンクの計算に使う最小周波数(メルスケール)
mlo = 0.0
# フィルタバンクの計算に使う最大周波数(メルスケール)
mhi = Mel(Nby2 + 1, fres)
# フィルタの数
numChans = 40
#numChans = 100
maxChan = numChans + 1

# フィルタバンクの初期化
def init_fbank():
    ms = mhi - mlo
    # 各フィルタ中心周波数を格納するarray
    cf = np.zeros((maxChan + 1))
    # 中心周波数の計算
    for chan in range(1, maxChan + 1):
        # 中心周波数はメルスケールで等間隔に並ぶ
        cf[chan] = (1.0 * chan / maxChan) * ms + mlo

    # 各周波数がどのフィルタの左半分に属するかを計算しておく
    loChan = np.zeros((Nby2 + 1), dtype=np.int)
    chan = 1
    for k in range(1, Nby2 + 1):
        # 範囲外の周波数はどのフィルタもかけない
        if k < klo or k > khi:
            loChan[k] = -1
        else:
            # ある周波数の対応するメル周波数
            melk = Mel(k, fres)
            # そのフィルタの中心周波数より大きくなる最初のフィルタを求める
            while (cf[chan] < melk) and (chan <= maxChan):
                chan = chan + 1
                if not (cf[chan] < melk and chan <= maxChan):
                    break
            # 当該周波数は、求めたフィルタの左半分に属する
            loChan[k] = chan - 1

    # 各周波数にかけられるフィルタ(三角形の左半分)の重みを計算しておく
    loWt = np.zeros((Nby2 + 1))
    for k in range(1, Nby2 + 1):
        chan = loChan[k]
        # 範囲外の周波数にはどのフィルタもかけられない
        if k < klo or k > khi:
            loWt[k] = 0.0
        else:
            # 直線の傾き(=フィルタの重み)を計算
            if chan > 0:
                loWt[k] = ((cf[chan + 1] - Mel(k, fres)) / (cf[chan + 1] - cf[chan]))
            else:
                loWt[k] = (cf[1] - Mel(k, fres)) / (cf[1] - mlo)

    return cf, loChan, loWt

def get_lmfb(cf, loChan, loWt, htk_ek):
    mfb = np.zeros((htk_ek.shape[0], numChans + 1))
    # 各周波数について、各フィルタのエネルギーに寄与する分を計算する
    for k in range(klo, khi + 1):
        ek = htk_ek[:, k]
        # その周波数が左半分に属するようなフィルタの番号
        bin = loChan[k]
        # その周波数のエネルギーにフィルタ重みをかけた値
        t1 = loWt[k] * ek
        if bin > 0:
            # 左半分については、正の傾きの重みがかけられる
            mfb[:, bin] += t1
        if bin < numChans:
            # 一つ上のフィルタについては、当該周波数はその右半分に属するので、負の傾きの重みがかけられる
            mfb[:, bin + 1] += ek - t1

    # 対数をとる
    return np.log(np.clip(mfb[:, 1:numChans+1], 1e-8, None))

def get_spec(signal):
    # 与えられた波形データが何フレームに対応するか計算しておく
    num_frames = int(np.ceil(float(np.abs(len(signal) - frame_length)) / frame_step))
    # 効率よく以降の特徴量の計算ができるように、最初にフレームごとの波形データに分けて(num_framesx400)のarrayに格納しておく
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    original_frames = signal[indices.astype(np.int32, copy=False)]
    # 高域強調を行う
    frames = np.hstack((original_frames[:, 0:1] * (1.0 - pre_emphasis), original_frames[:, 1:] - pre_emphasis * original_frames[:, :-1]))
    # ハミング窓をかける
    frames *= np.hamming(frame_length)
    # FFTをした上で絶対値を取り、絶対値スペクトルを計算する
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    htk_ek = np.hstack((mag_frames[:, 0:1], mag_frames[:, 0:256]))
    return htk_ek

class TedliumPreprocess_lmfb(Dataset):                                                     
    """
    A preprocess for Tedlium datasets

    """                                                   
    def __init__(self, segments_file, save_dir=None):                                     
        """                                                                     
        Args:                                                                   
            csv_file (string): Path to the csv file with annotations.           
            root_dir (string): Directory with all the wavs.                     
                                                                                
        """                                                                     
        self.segments_landmarks = pd.read_csv(segments_file, sep=' ', header=None)

        self.save_dir_wav = os.path.join(save_dir, 'wav')
        self.save_dir_lmfb = os.path.join(save_dir, 'lmfb')
        self.cf, self.loChan, self.loWt = init_fbank()

    def load_wav(self, filename):             
        #  sample_rate, signal = scipy.io.wavfile.read(load_file)                                #   
        return scipy.io.wavfile.read(filename)

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
                                                                                
    def __len__(self):                                                          
        return len(self.segments_landmarks)                                        
                                                                                
    def __getitem__(self, idx): 
        save_id, file_id, _, _ = self.segments_landmarks.loc[idx, :]
        dir_name = os.path.join(self.save_dir_wav, file_id)

        wav_name = os.path.join(dir_name, save_id+'.wav')

        save_dir = os.path.join(self.save_dir_lmfb, file_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        save_name = os.path.join(save_dir, save_id+'.npy')

        sample_rate, signal = self.load_wav(wav_name)

        lmfb = get_lmfb(self.cf, self.loChan, self.loWt, get_spec(signal))

        np.save(save_name, lmfb)

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
    return TedliumPreprocess_lmfb('/n/work1/ueno/data/tedlium3/data/train/segments', '/n/work1/ueno/data/tedlium3/') 

def _pad_mel(inputs):
    _pad = 0
    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(x, [[0,max_len - mel_len],[0,0]], mode='constant', constant_values=_pad)
    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])

if __name__ == '__main__':
    datasets = get_dataset()
    dataloader = DataLoader(datasets, batch_size=1, drop_last=False, num_workers=8)
    from tqdm import tqdm
    pbar = tqdm(dataloader)
    for d in pbar:
        pass
