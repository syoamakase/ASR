# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from Models.decoder import Decoder
from Models.encoder import Encoder, WaveEncoder
from Models.Transformer_encoder import TransformerEncoder

class RNNTModel(nn.Module):
    def __init__(self, hp):
        super(AttModel, self).__init__()
        self.hp = hp
        if hp.encoder_type == 'Wave':
            self.encoder = WaveEncoder()
        elif hp.encoder_type == 'Conformer':
            self.encoder = TransformerEncoder(hp.lmfb_dim, 256, 16, 4, 0.1)
        else:
            self.encoder = Encoder(hp)
        self.decoder = Decoder(hp)

    def forward(self, x, lengths, targets, src_pos=None):
        if self.hp.encoder_type == 'Conformer':
            hbatch = self.encoder(x, src_pos)
        else:
            hbatch = self.encoder(x, lengths)
        youtput = self.decoder(hbatch, lengths, targets)

        return youtput
    
    def decode(self, x, lengths, model_lm, src_pos):
        with torch.no_grad():
            if self.hp.encoder_type == 'Conformer':
                hbatch = self.encoder(x, src_pos)
            else:
                hbatch = self.encoder(x, lengths)
            results = self.decoder.decode(hbatch, lengths, model_lm)

        return results
    
    def decode_v2(self, x, lengths, model_lm, src_pos):
        with torch.no_grad():
            if self.hp.encoder_type == 'Conformer':
                hbatch = self.encoder(x, src_pos)
            else:
                hbatch = self.encoder(x, lengths)
            results = self.decoder.decode_v2(hbatch, lengths, model_lm)

        return results
