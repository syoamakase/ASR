# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from Models.encoder import Encoder, CNN_Encoder

import hparams as hp

class CTCModel(nn.Module):
    def __init__(self):
        super(CTCModel, self).__init__()
        if hp.encoder_type == 'CNN':
            self.encoder = CNN_Encoder()
        else:
            self.encoder = Encoder()
        
        self.decoder = nn.Linear(hp.num_hidden_nodes * 2, hp.num_classes + 1)

    def forward(self, x, lengths, targets):
        hbatch = self.encoder(x, lengths)
        youtput = self.decoder(hbatch)

        return youtput
    
    def decode(self, x, lengths):
        with torch.no_grad():
            hbatch = self.encoder(x, lengths)
            results = self.decoder(hbatch)

        return results
