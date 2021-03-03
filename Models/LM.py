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
