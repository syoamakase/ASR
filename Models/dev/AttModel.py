# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from Models.decoder import Decoder
from Models.encoder import Encoder, WaveEncoder
from Models.dev.text_encoder import TextEncoder
from Models.Transformer_encoder import TransformerEncoder

class AttModelWithText(nn.Module):
    def __init__(self, hp):
        super(AttModelWithText, self).__init__()
        self.hp = hp
        text_dim = False
        assert False, f'please fill {text_dim}!!!!'

        self.text_encoder = TextEncoder(text_dim, hp)
        self.encoder = Encoder(hp)
        self.decoder = Decoder(hp)

    def forward(self, x, lengths, targets, text=False, src_pos=None):
        if text:
            hbatch = self.text_encoder(x, src_pos)
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
