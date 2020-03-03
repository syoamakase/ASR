# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from struct import unpack, pack
import copy
import math

from Datasets import dataset_transformer
from torch.utils.data import DataLoader
import hparams as hp

def load_dat(filename):
    fh = open(filename, "rb")
    spam = fh.read(12)
    nSamples, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)
    veclen = int(sampSize / 4)
    fh.seek(12, 0)
    dat = np.fromfile(fh, dtype=np.float32)
    dat = dat.reshape(int(len(dat) / veclen), veclen)
    dat = dat.byteswap()
    fh.close()
    return dat

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(-0.1, 0.1)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)
    if classname.find('LSTM') != -1:
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            if 'bias' in name:
                param.data.fill_(0)
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 1500, dropoutrate = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropoutrate)
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropoutrate = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropoutrate)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask = None):
        batch_size = q.size(0)

        q = self.W_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # (B, heads, T, d_k)
        k = self.W_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # (B, heads, T, d_k)
        v = self.W_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # (B, heads, T, d_k)
        weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) # (B, heads, T(q), T(k))
        if mask is not None:
            # normal mh attention: mask: (B, 1, T(k)) -> (B, 1, 1, T(k))
            # masked mh attention: mask: (B, T(q), T(k)) -> (B, 1, T(q), T(k))
            mask = mask.unsqueeze(1)
            weights = weights.masked_fill(mask == 0, -1e9)

        weights = F.softmax(weights, dim=-1) # sum taken along the T(k) axis is 1
        weights = self.dropout(weights) # ???
        glimpse_heads = torch.matmul(weights, v) # (B, heads, T, d_k)
        glimpse_heads = glimpse_heads.transpose(1, 2) # (B, T, heads, d_k)
        concat_output = glimpse_heads.contiguous().view(batch_size, -1, self.d_model) # (B, T, d_model)
        output = self.out(concat_output) # ???
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropoutrate = 0.1):
        super().__init__() 
        self.W1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropoutrate)
        self.W2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.W1(x))) # (B, T, d_model) -> (B, T, 2048)
        x = self.W2(x) # (B, T, 2048) -> (B, T, d_model)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropoutrate=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(n_heads, d_model, dropoutrate = dropoutrate)
        self.ff = FeedForward(d_model, dropoutrate = dropoutrate)
        self.dropout_1 = nn.Dropout(dropoutrate)
        self.dropout_2 = nn.Dropout(dropoutrate)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
class Encoder(nn.Module):
    def __init__(self, d_feature, d_model, n_layers, n_heads, dropoutrate = 0.1):
        super().__init__()
        self.n_layers = n_layers
        self.input_lin = nn.Linear(d_feature, d_model)
        self.pe = PositionalEncoder(d_model, dropoutrate = dropoutrate)
        self.layers = get_clones(EncoderLayer(d_model, n_heads, dropoutrate), n_layers)
        self.norm = Norm(d_model)

    def forward(self, src, input_mask):
        x = self.input_lin(src)
        x = self.pe(x)
        for i in range(self.n_layers):
            x = self.layers[i](x, input_mask)
            # downsampling
            # if i % 3 == 2:
            #     x = x[:, ::2, :]
            #     input_mask = input_mask[:, :, ::2]
            
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropoutrate=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropoutrate)
        self.dropout_2 = nn.Dropout(dropoutrate)
        self.dropout_3 = nn.Dropout(dropoutrate)
        
        self.attn_1 = MultiHeadAttention(n_heads, d_model, dropoutrate=dropoutrate)
        self.attn_2 = MultiHeadAttention(n_heads, d_model, dropoutrate=dropoutrate)
        self.ff = FeedForward(d_model, dropoutrate=dropoutrate)

    def forward(self, x, enc_output, input_mask, label_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, label_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, enc_output, enc_output, input_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, dropoutrate):
        super().__init__()
        self.n_layers = n_layers
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropoutrate=dropoutrate)
        self.layers = get_clones(DecoderLayer(d_model, n_heads, dropoutrate), n_layers)
        self.norm = Norm(d_model)
    def forward(self, enc_output, label, input_mask, label_mask):
        x = self.embed(label)
        x = self.pe(x)
        for i in range(self.n_layers):
            x = self.layers[i](x, enc_output, input_mask, label_mask)
        return self.norm(x)





class InputCNN(nn.Module):
    def __init__(self, num_filters, pooling_size, dropoutrate):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, 2)
    def forward(self, speech, maxlen):
        # NxTxF -> NxCxTxF
        x = speech.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # NxCxT -> NxTxC
        b, c, t, f = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)

        x = x[:, :maxlen, :]
        return x



    
class Transformer(nn.Module):
    def __init__(self, d_feature, vocab_size, d_model, n_enc_layers, n_dec_layers, n_heads, dropoutrate):
        super().__init__()



        
        self.cnn = InputCNN(32, 2, dropoutrate)

        


        #self.encoder = Encoder(d_feature, d_model, n_enc_layers, n_heads, dropoutrate)
        #self.encoder = Encoder(32, d_model, n_enc_layers, n_heads, dropoutrate)
        #### should be modified
        self.encoder = Encoder(288, d_model, n_enc_layers, n_heads, dropoutrate)



        self.decoder = Decoder(vocab_size, d_model, n_dec_layers, n_heads, dropoutrate)
        self.out = nn.Linear(d_model, vocab_size)
    def forward(self, speech, label, input_mask, label_mask, downsampled_maxlen):
        ##### downsampling using 2 conv layers
        speech_downsampled = self.cnn(speech, downsampled_maxlen)
        #####
        #enc_output = self.encoder(speech, input_mask)



        enc_output = self.encoder(speech_downsampled, input_mask)
        dec_output = self.decoder(enc_output, label, input_mask, label_mask)
        output = self.out(dec_output)
        return output

def create_input_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, dtype=torch.int32).expand(len(lengths), max_len).to(DEVICE) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-2)
    return mask

def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)
    return np_mask.to(DEVICE)

def create_label_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, dtype=torch.int32).expand(len(lengths), max_len).to(DEVICE) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-2)
    np_mask = nopeak_mask(max_len)
    mask = mask & np_mask
    return mask

def spec_augment(x):
    aug_f = np.random.randint(0, aug_F)
    aug_f0 = np.random.randint(0, LMFB_DIM - aug_f)
    aug_f_mask_from = aug_f0
    aug_f_mask_to = aug_f0 + aug_f
    x[:, aug_f_mask_from:aug_f_mask_to] = 0.0
    if x.shape[0] > aug_T:
        aug_t = np.random.randint(0, aug_T)
        aug_t0 = np.random.randint(0, x.shape[0] - aug_t)
        aug_t_mask_from = aug_t0
        aug_t_mask_to = aug_t0 + aug_t
        x[aug_t_mask_from:aug_t_mask_to, :] = 0.0
    return x

def load_model(model_file):
    model_state = torch.load(model_file)
    is_multi_loading = True if torch.cuda.device_count() > 1 else False
    # This line may include bugs!!
    is_multi_loaded = True if 'module' in list(model_state.keys())[0] else False

    if is_multi_loaded is is_multi_loading:
        return model_state

    elif is_multi_loaded is False and is_multi_loading is True:
        new_model_state = {}
        for key in model_state.keys():
            new_model_state['module.'+key] = model_state[key]

        return new_model_state

    elif is_multi_loaded is True and is_multi_loading is False:
        new_model_state = {}
        for key in model_state.keys():
            new_model_state[key[7:]] = model_state[key]

        return new_model_state

    else:
        print('ERROR in load model')
        sys.exit(1)


def frame_stacking(x):
    newlen = len(x) // 3
    stacked_x = x[0:newlen * 3].reshape(newlen, 40 * 3)
    return stacked_x

def get_learning_rate(step, d_model):
    return warmup_factor * min(step ** -0.5, step * warmup_step ** -1.5) * (d_model ** -0.5)

def train_epoch(model, optimizer, training_data, total_step, d_model):

    step = total_step
    acc_utts = 0
    
    datasets = dataset_transformer.get_dataset(hp.script_file)
    collate_fn_transformer = dataset_transformer.collate_fn_transformer
    if hp.batch_size is not None:
        sampler = dataset_transformer.NumBatchSampler(datasets, hp.batch_size)
    elif hp.max_seqlen is not None:
        sampler = dataset_transformer.LengthsBatchSampler(datasets, hp.max_seqlen, hp.lengths_file)

    dataloader = DataLoader(datasets, batch_sampler=sampler, num_workers=4, collate_fn=collate_fn_transformer)
    for d in dataloader:
        lr = get_learning_rate(step, d_model)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        text, mel_input, pos_text, pos_mel, text_lengths = d

        text = text.to(DEVICE)
        mel_input = mel_input.to(DEVICE)
        pos_text = pos_text
        pos_mel = pos_mel
        text_lengths = text_lengths

        batch_size = mel_input.shape[0]

        xs_lengths = torch.tensor(np.array([x for x in pos_mel.max(dim=1)[0]], dtype=np.int32), device = DEVICE)
        ts_lengths = torch.tensor(np.array([t for t in pos_text.max(dim=1)[0]], dtype=np.int32), device = DEVICE)

        # padded_xs = nn.utils.rnn.pad_sequence(xs, batch_first = True) # NxTxD
        # padded_ts = nn.utils.rnn.pad_sequence(ts, batch_first = True) # NxT

        padded_ts_input = text[:, :-1]
        padded_ts_target = text[:, 1:]
        ts_lengths_input = ts_lengths - 1
        
        #### should be modified
        downsampled_maxlen = (xs_lengths // 4).max() - 1
        ####### downsampling using 2 conv layers
        #input_mask = create_input_mask(xs_lengths)
        input_mask = create_input_mask(xs_lengths // 4 - 1)
        #######

        label_mask = create_label_mask(ts_lengths_input)

        prediction = model(mel_input, padded_ts_input, input_mask, label_mask, downsampled_maxlen.item())

        loss = 0.0
        for i in range(batch_size):
            num_labels = ts_lengths_input[i]
            label = padded_ts_target[i, :num_labels]
            onehot_target = torch.zeros((len(label), NUM_CLASSES), dtype=torch.float32, device=DEVICE)
            for j in range(len(label)):
                onehot_target[j][label[j]] = 1.0
            ls_target = 0.9 * onehot_target + ((1.0 - 0.9) / (NUM_CLASSES - 1)) * (1.0 - onehot_target)
            loss += -(F.log_softmax(prediction[i][:num_labels], dim=1) * ls_target).sum()

        n_correct = 0
        for i in range(len(text)):
            num_labels = ts_lengths_input[i]
            label = padded_ts_target[i, :num_labels]
            tmp = prediction[i, :num_labels, :].max(1)[1]
            for j in range(num_labels):
                if tmp[j] == label[j]:
                    n_correct = n_correct + 1
        den = ts_lengths_input.sum().item()
        acc = 1.0 * n_correct / (1.0 * den)
        print('acc =', acc)

        print('lr =', lr)
        print('loss =', loss.item())
        sys.stdout.flush()
        optimizer.zero_grad()
        loss.backward()
        if CLIPGRAD:
            clip = 5.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        loss.detach()

        step = step + 1
    return step
        
if __name__ == '__main__':

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #feature_format = 'htk'
    feature_format = 'npy'
    LMFB_DIM = 40
    #CLIPGRAD = True
    CLIPGRAD = False
    SPECAUGMENT = True
    #SPECAUGMENT = False
    aug_F = 15
    aug_T = 20
    #BATCH_SIZE = 40
    BATCH_SIZE = 120
    NUM_CLASSES = int(sys.argv[1])
    script_file = sys.argv[2]
    save_dir = sys.argv[3]

    print('NUM_CLASSES =', NUM_CLASSES)
    print('script_file =', script_file)
    print('save_dir =', save_dir)
    print('SPECAUGMENT =', SPECAUGMENT)
    print('CLIPGRAD =', CLIPGRAD)
    print('PID =', os.getpid())
    print('HOST =', os.uname()[1])

    NUM_ENC_LAYERS = 12
    NUM_DEC_LAYERS = 6
    NUM_HEADS = 4
    DIM_MODEL = 256
    
    model = Transformer(d_feature = LMFB_DIM * 3, vocab_size = NUM_CLASSES, d_model = DIM_MODEL, n_enc_layers = NUM_ENC_LAYERS, n_dec_layers = NUM_DEC_LAYERS, n_heads = NUM_HEADS, dropoutrate = 0.1).to(DEVICE)
    model.apply(init_weight)
    model.train()

    # multi GPU
    model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True
    
    warmup_step = 25000
    warmup_factor = 2.0
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0, betas=(0.9, 0.98), weight_decay=1e-9)
    os.makedirs(save_dir, exist_ok=True)

    training_data = [line for line in open(script_file)]

    #step = 1
    step = 38200
    model.load_state_dict(torch.load('model.multi.e12.d6.h4.d256.specaug.dataset/network.epoch100'))
    optimizer.load_state_dict(torch.load('model.multi.e12.d6.h4.d256.specaug.dataset/network.optimizer.epoch100'))
    
    for epoch in range(100, 220):

        step = train_epoch(model, optimizer, training_data, step, DIM_MODEL)
    
        #torch.save(model.state_dict(), save_dir+"/network.epoch{}".format(epoch+1))
        if (epoch+1) > 50:
            torch.save(model.module.state_dict(), save_dir+"/network.epoch{}".format(epoch+1))
            torch.save(optimizer.state_dict(), save_dir+"/network.optimizer.epoch{}".format(epoch+1))
        print("EPOCH {} end".format(epoch+1))
