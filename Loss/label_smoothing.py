# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import hparams as hp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class LabelSmoothingLoss(nn.Module):
#     def __init__(self):
#         super(LabelSmoothingLoss, self).__init__()

def label_smoothing_loss(predicted_label, target, text_lengths, eps=0.1):
    loss = 0.0
    B, T, L = predicted_label.shape
    log_prob = F.log_softmax(predicted_label, dim=2)
    onehot = torch.zeros((B * T, L)).cuda().scatter(1, target.view(-1, 1), 1)
    onehot = onehot * (1 - eps) + (1 - onehot) * eps / (L - 1)
    onehot = onehot.view(B, T, L)
    for i, t in enumerate(text_lengths):
        loss += -(onehot[i, :t, :] * log_prob[i, :t, :]).sum() / t
    #loss /= B
    return loss

def label_smoothing_loss_legacy(y_pred, y):
    loss = -(F.log_softmax(y_pred, dim=1) * y).sum()
    return loss
