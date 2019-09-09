# -*- coding: utf-8 -*-
"""
The implementation of calculate log mel-scale filter bank like HTK.

python_speech_features(https://github.com/jameslyons/python_speech_features) exists
however the results of it is differnent from these of HTK.
"""

import argparse
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
from struct import unpack, pack
import warnings

def Mel(k, fresh):
    return 1127 * np.log(1 + (k - 1) * fres)

# configuration
pre_emphasis = 0.97
frame_length = 400
frame_step = 160
NFFT = 512
Nby2 = int(NFFT / 2)
fres = 16000 / (NFFT * 700)
klo = 2
khi = Nby2
mlo = 0.0
mhi = Mel(Nby2 + 1, fres)
numChans = 40
# numChans = 100
maxChan = numChans + 1


def save_dat_HTK_format(dat_cpu, out_file):
    """Save file as HTK format.
    However, zero padding is added (BUG!!)

    Args:
      dat_cpu: Lmfb data.
      out_file : The output filename.
    """
    warnings.warn("save_dat_HTK_format is called. The results are not the same of these of HTK strictly.")
    veclen = dat_cpu.shape[0]
    nSamples = dat_cpu.shape[1]
    sampPeriod = 100000
    sampSize = veclen * 4
    parmKind = 838
    fh = open(out_file, "wb")
    fh.seek(0,0)
    fh.write(pack(">IIHH", nSamples, sampPeriod, sampSize, parmKind))
    for raw in dat_cpu.T:
        r = np.array(raw, dtype=np.float32).byteswap()
        fh.write(pack("<%sf" % r.size, *r.flatten('F')))
        # np.array(raw, dtype=np.float32).byteswap().tofile(fh)
    fh.close()

def init_fbank():
    ms = mhi - mlo;
    cf = np.zeros((maxChan + 1))
    for chan in range(1, maxChan + 1):
        cf[chan] = (1.0 * chan / maxChan) * ms + mlo

    # create lochan map
    loChan = np.zeros((Nby2 + 1), dtype=np.int)
    chan = 1
    for k in range(1, Nby2 + 1):
        if k < klo or k > khi:
            loChan[k] = -1
        else:
            melk = Mel(k, fres)
            while (cf[chan] < melk) and (chan <= maxChan):
                chan = chan + 1
                if not (cf[chan] < melk and chan <= maxChan):
                    break
            loChan[k] = chan - 1

    loWt = np.zeros((Nby2 + 1))
    for k in range(1, Nby2 + 1):
        chan = loChan[k]
        if k < klo or k > khi:
            loWt[k] = 0.0
        else:
            if chan > 0:
                loWt[k] = ((cf[chan + 1] - Mel(k, fres)) / (cf[chan + 1] - cf[chan]))
            else:
                loWt[k] = (cf[1] - Mel(k, fres)) / (cf[1] - mlo)

    return cf, loChan, loWt

def get_lmfb(cf, loChan, loWt, htk_ek):
    mfb = np.zeros((htk_ek.shape[0], numChans + 1))
    # mfb = np.zeros((htk_ek.shape[0], numChans))

    for k in range(klo, khi + 1):
        ek = htk_ek[:, k]
        bin = loChan[k]
        t1 = loWt[k] * ek
        if bin > 0:
            mfb[:, bin] += t1
        if bin < numChans:
            mfb[:, bin + 1] += ek - t1
            # mfb[:, bin] += ek - t1

    # return np.log(mfb[:, 1:maxChan])
    return np.log(np.clip(mfb[:, 1:numChans+1], 1e-8, None))


def get_spec(signal):
    num_frames = int(np.ceil(float(np.abs(len(signal) - frame_length)) / frame_step))
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    original_frames = signal[indices.astype(np.int32, copy=False)]
    # preemp
    frames = np.hstack((original_frames[:, 0:1] * (1.0 - pre_emphasis), original_frames[:, 1:] - pre_emphasis * original_frames[:, :-1]))
    # hamming
    frames *= np.hamming(frame_length)
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    htk_ek = np.hstack((mag_frames[:, 0:1], mag_frames[:, 0:256]))
    return htk_ek

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wav2lmfb_file')
    args = parser.parse_args()

    wav2lmfb_file = args.wav2lmfb_file
    cf, loChan, loWt = init_fbank()

    with open(wav2lmfb_file) as f:
        for line in f:
            load_file, save_file = line.strip().split(' ')
            # for each utt
            sample_rate, signal = scipy.io.wavfile.read(load_file)  # File assumed to be in the same directory
    
            lmfb = get_lmfb(cf, loChan, loWt, get_spec(signal))
            
            np.save(save_file, lmfb)
            #save_dat_HTK_format(lmfb.T, "tmp-python.htk")
