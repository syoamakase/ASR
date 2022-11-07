# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_lm(nn.Module):
    def __init__(self, hp_LM):
        super(Model_lm, self).__init__()

        self.embeddings = nn.Embedding(hp_LM.num_classes, hp_LM.num_hidden_LM)
        self.lstm = nn.LSTM(input_size=hp_LM.num_hidden_LM, hidden_size=hp_LM.num_hidden_LM, num_layers=4, dropout=0.2, batch_first=True)
        self.linear = nn.Linear(hp_LM.num_hidden_LM, hp_LM.num_classes)

    def forward(self, input_):
        embeds = self.embeddings(input_)

        lstm_out, (_, _) = self.lstm(embeds)
        prediction = self.linear(lstm_out)

        return prediction

class Model_lm_old(nn.Module):
    def __init__(self):
        super(Model_lm_old, self).__init__()
        self.vocab_size = 34305
        self.vocab_base = 34331
        self.embeddings = nn.Embedding(self.vocab_size+1, 512)
        self.lstm = nn.LSTM(input_size = 512, hidden_size = 256, num_layers = 3, dropout = 0.0, batch_first = True)
        self.linear = nn.Linear(256, self.vocab_size)


    def forward(self, input):
        batch_size = input.size(0)
        input_length = input.size(1)

        embeds = self.embeddings(input)

        lstm_out, (hy, cy) = self.lstm(F.dropout(embeds, p = 0.0))
        prediction = self.linear(lstm_out)
        prediction = F.log_softmax(prediction, dim=2)

        results = torch.zeros((batch_size, input_length, self.vocab_base), device=input.device).fill_(prediction.min())
        results[:,:,1:self.vocab_size] = prediction[:, :, 1:]
        # results[:,:,:] = prediction[:, :, :]

        return results

