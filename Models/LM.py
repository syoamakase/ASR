# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

#vocab_size = 34305
#vocab_base = 34331
vocab_size = 6027
vocab_base = 6027

class Model_lm(nn.Module):
    def __init__(self):
        super(Model_lm, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, 512)
        self.lstm = nn.LSTM(input_size = 512, hidden_size = 256, num_layers = 3, dropout = 0.0, batch_first = True)
        self.linear = nn.Linear(256, vocab_size)

    def forward(self, input):
        batch_size = input.size(0)
        input_length = input.size(1)

        embeds = self.embeddings(input)

        lstm_out, (hy, cy) = self.lstm(F.dropout(embeds, p = 0.0))
        prediction = self.linear(lstm_out)
        prediction = F.log_softmax(prediction, dim=2)

        results = torch.zeros((batch_size, input_length, vocab_base), device=input.device).fill_(prediction.min())
        #results[:,:,1:vocab_size] = prediction[:, :, 1:]
        results[:,:,:] = prediction[:, :, :]

        return results
