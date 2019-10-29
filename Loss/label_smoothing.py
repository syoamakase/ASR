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

def label_smoothing_loss(predict_ts, ls_target):
    loss = -(F.log_softmax(predict_ts, dim=1) * ls_target).sum()
    return loss
