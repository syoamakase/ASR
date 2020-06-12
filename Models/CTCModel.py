# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from Models.encoder import Encoder, WaveEncoder

from utils import hparams as hp

class CTCModel(nn.Module):
    def __init__(self):
        super(CTCModel, self).__init__()
        if hp.encoder_type == 'Wave':
            self.encoder = WaveEncoder()
        else:
            self.encoder = Encoder()
        
        self.decoder = nn.Linear(hp.num_hidden_nodes * 2, hp.num_classes + 1)

    def forward(self, x, lengths, targets):
        hbatch = self.encoder(x, lengths)
        youtput = self.decoder(hbatch)

        return youtput
    
    def decode(self, x, lengths, model_lm=None):
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
            
    def align(self, x, lengths, target):
        with torch.no_grad():
            hbatch = self.encoder(x, lengths)
            # (B, T, V)
            decoder_output = self.decoder(hbatch)
            print(decoder_output)

        return results