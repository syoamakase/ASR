import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

#import utils.hparams as hp
from utils import hparams as hp
from Models.AttModel import AttModel
from utils.utils import load_model, fill_variables

class WaveModel(nn.Module):
    def __init__(self):
        super(WaveModel, self).__init__()
        # waveform
        self.preemp = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, bias=False)
        tmp = np.zeros((1, 1, 2))
        tmp[:, :] = np.array([-0.97, 1.0])
        K = torch.Tensor(tmp)
        self.preemp.weight.data = K

        #self.comp = nn.Conv1d(in_channels=1, out_channels=hp.window_dim, kernel_size=hp.window_size, stride=1, padding=0, bias=False)
        self.comp = nn.Conv1d(in_channels=1, out_channels=80, kernel_size=800, stride=1, padding=0, bias=False)

        nn.init.kaiming_normal_(self.comp.weight.data)

        tmp = np.zeros((80, 1, 800))
        #tmp[:, :] = scipy.hanning(800 + 1)[:-1]
        #self.tmp = tmp * tmp

        self.instancenorm = nn.InstanceNorm1d(80)

        # encoder
        self.bi_lstm = nn.LSTM(input_size=80 * hp.frame_stacking, hidden_size=hp.num_hidden_nodes, num_layers=5, batch_first=True, dropout=0.2, bidirectional=True)
        # attention
        self.L_se = nn.Linear(hp.num_hidden_nodes, hp.num_hidden_nodes * 2, bias=False)
        self.L_he = nn.Linear(hp.num_hidden_nodes * 2, hp.num_hidden_nodes * 2)
        self.L_ee = nn.Linear(hp.num_hidden_nodes * 2, 1, bias=False)
        # conv attention
        self.F_conv1d = nn.Conv1d(1, 10, 100, stride=1, padding=50, bias=False)
        self.L_fe = nn.Linear(10, hp.num_hidden_nodes*2, bias=False)

        # decoder
        self.L_sy = nn.Linear(hp.num_hidden_nodes, hp.num_hidden_nodes, bias=False)
        self.L_gy = nn.Linear(hp.num_hidden_nodes * 2, hp.num_hidden_nodes)
        self.L_yy = nn.Linear(hp.num_hidden_nodes, hp.num_classes)

        self.L_ys = nn.Embedding(hp.num_classes, hp.num_hidden_nodes * 4)
        self.L_ss = nn.Linear(hp.num_hidden_nodes, hp.num_hidden_nodes * 4, bias=False)
        self.L_gs = nn.Linear(hp.num_hidden_nodes * 2, hp.num_hidden_nodes * 4)


def load_waveform_model(model, waveform_path):
    hp_path = os.path.join(os.path.dirname(waveform_path), 'hparams.py')
    wave_model = WaveModel()
    wave_model.load_state_dict(load_model(waveform_path))
    wave_model_dict = wave_model.state_dict()
    loaded_parameter = {}
    model_dict = model.state_dict()
    for k, v in wave_model_dict.items():
        if 'bi_lstm' in k:
            loaded_parameter['encoder.'+k] = v
        elif 'L_se' in k or 'L_he' in k or 'L_ee' in k or 'L_fe' in k or 'F_conv1d' in k:
            loaded_parameter['decoder.att.'+k] = v
        elif 'L_' in k:
            loaded_parameter['decoder.'+k] = v

    model_dict.update(loaded_parameter)
    model.load_state_dict(model_dict)
    print(f'loaded {waveform_path} wave model')
    import pdb; pdb.set_trace()
    return model

if __name__ == '__main__':
    #hp.configure('/n/rd24/ueno/trial_exp/waveform/model.librispeech460_ted2tts.waveform.lowpass_constant.framestacking.direct80_abs_shift200_window800.no_normalize.emb/hparams.py')
    hp.configure('checkpoints.librispeech.train_460_ted2_tts.wav_feats.1kbpe/hparams.py')

    fill_variables(hp)
    model = AttModel(hp)

    load_waveform_model(model, '/n/rd24/ueno/trial_exp/waveform/model.librispeech460_ted2tts.waveform.lowpass_constant.framestacking.direct80_abs_shift200_window800.no_normalize.emb/finetune_all/network.epoch10')
