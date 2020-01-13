# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from Models.encoder import Encoder, CNN_Encoder, WaveEncoder

#import hparams as hp
from utils import hparams as hp

class CTCModel(nn.Module):
    def __init__(self):
        super(CTCModel, self).__init__()
        if self.hp.encoder_type == 'CNN':
            self.encoder = CNN_Encoder()
        elif self.hp.encoder_type == 'Wave':
            self.encoder = WaveEncoder()
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
            decoder_output = self.decoder(hbatch)

            results = []
            prev_id = hp.num_classes + 1
            for x in decoder_output[0].argmax(dim=1):
                if int(x) != prev_id and int(x) != hp.num_classes:
                    results.append(int(x))
                prev_id = int(x)

        return results

    def analyze(self, x, lengths):
        with torch.no_grad():
            hbatch = self.encoder(x, lengths)
            decoder_output = self.decoder(hbatch)

            results = []
            prev_id = hp.num_classes + 1
            for x in decoder_output[0].argmax(dim=1):
                results.append(int(x))

        return results
            
