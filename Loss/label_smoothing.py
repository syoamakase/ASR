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

def label_smoothing_loss(predict_ts, ts):
    onehot_target = torch.eye(hp.num_classes)[ts].to(DEVICE)
    ls_target = 0.9 * onehot_target + ((1.0 - 0.9) / (hp.num_classes - 1)) * (1.0 - onehot_target)
    loss = -(F.log_softmax(predict_ts, dim=1) * ls_target).sum()
    return loss
