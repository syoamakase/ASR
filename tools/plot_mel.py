import numpy as np
import torch
import matplotlib.pyplot as plt
from struct import pack, unpack
import sys

def load_dat(filename):
    """
    To read binary data in htk file.
    The htk file includes log mel-scale filter bank.

    Args:
        filename : file name to read htk file

    Returns:
        dat : 120 (means log mel-scale filter bank) x T (time frame)

    """
    fh = open(filename, "rb")
    spam = fh.read(12)
    nSamples, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)
    veclen = int(sampSize / 4)
    fh.seek(12, 0)
    dat = np.fromfile(fh, dtype=np.float32)
    dat = dat.reshape(int(len(dat) / veclen), veclen)
    dat = dat.byteswap()
    fh.close()
    return dat[:,:40]

if __name__ == '__main__':
    argc = len(sys.argv)

    for i in range(1,argc):
        filename = sys.argv[i]
        if '.npy' in filename:
            data = np.load(filename)
        elif '.htk' in filename:
            data = load_dat(filename).T
        elif '.pt' in filename:
            data = torch.load(filename).numpy()
        #data.reshape(-1, 80)
        if i == 1:
            length = data.shape[1]
        plt.rcParams["font.size"] = 15
        plt.rcParams['figure.figsize'] = 15,10
        plt.subplot(argc-1, 1, i)
        #plt.title(sys.argv[i])
        plt.xlim(0,length)
        if i == 2:
            plt.ylabel('frequency')
        #plt.yticks(yyy+0.5, yyy+1, size=13)
        if i == 3:
            plt.xlabel('time frame')
        #plt.tick_params(labelbottom=False,
        #        labelleft=False,
        #        labelright=False,
        #        labeltop=False)
        #plt.tick_params(bottom=False,
        #        left=False,
        #        right=False,
        #        top=False)
        plt.imshow(data, origin='lower')
    
    plt.show()
