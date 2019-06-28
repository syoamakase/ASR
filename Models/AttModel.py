# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from Models.decoder import Decoder
from Models.encoder import Encoder, CNN_Encoder, WaveEncoder

import hparams as hp

class AttModel(nn.Module):
    def __init__(self):
        super(AttModel, self).__init__()
        if hp.encoder_type == 'CNN':
            self.encoder = CNN_Encoder()
        elif hp.encoder_type == 'Wave':
            self.encoder = WaveEncoder()
        else:
            self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, lengths, targets):
        hbatch = self.encoder(x, lengths)
        youtput = self.decoder(hbatch, lengths, targets)

        return youtput
    
    def decode(self, x, lengths):
        with torch.no_grad():
            hbatch = self.encoder(x, lengths)
            results = self.decoder.decode(hbatch, lengths)

        return results
