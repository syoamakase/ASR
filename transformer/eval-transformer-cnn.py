# -*- coding: utf-8 -*-

import sys
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from operator import itemgetter, attrgetter
from struct import unpack, pack
import math
import copy

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
#    def __init__(self, d_model, max_seq_len = 200, dropout = 0.1):
#    def __init__(self, d_model, max_seq_len = 3000, dropoutrate = 0.1):
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
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)

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

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
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

    def forward(self, x, enc_output, input_mask = None, label_mask = None):
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
    def forward(self, enc_output, label, input_mask = None, label_mask = None):
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
    def forward(self, speech, maxlen = None):
        # NxTxF -> NxCxTxF

        x = speech.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # NxCxT -> NxTxC                                                                                                                        
        b, c, t, f = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        if maxlen is not None:
            x = x[:, :maxlen, :]

        return x



class Transformer(nn.Module):
    def __init__(self, d_feature, vocab_size, d_model, n_enc_layers, n_dec_layers, n_heads, dropoutrate):
        super().__init__()



        self.cnn = InputCNN(32, 2, dropoutrate)


        #self.encoder = Encoder(d_feature, d_model, n_enc_layers, n_heads, dropoutrate)
        #### should be modified                                                                                                                 
        self.encoder = Encoder(288, d_model, n_enc_layers, n_heads, dropoutrate)




        self.decoder = Decoder(vocab_size, d_model, n_dec_layers, n_heads, dropoutrate)
        self.out = nn.Linear(d_model, vocab_size)
    def forward(self, speech, label = None, input_mask = None, label_mask = None):
        sos_id = 0
        #sos_id = 1
        token_finalist = []


        
        #speech_downsampled = self.cnn(speech, downsampled_maxlen)
        speech_downsampled = self.cnn(speech)
        #enc_output = self.encoder(speech, input_mask)
        enc_output = self.encoder(speech_downsampled, input_mask)




        hypes = [([sos_id], 0.0)]
        for _out_step in range(200):
            new_hypes = []
            for hype in hypes:
                out_seq, seq_score = hype
                out_seq_torch = torch.from_numpy(np.array(out_seq)).to(DEVICE).unsqueeze(0)
                lengths = torch.from_numpy(np.array([len(out_seq)])).to(DEVICE)
                label_mask = create_label_mask(lengths)
                dec_output = self.decoder(enc_output, out_seq_torch, label_mask = label_mask)
                output = self.out(dec_output)
                scores = F.softmax(output[:, -1, :], dim=1).data.squeeze(0)
                #scores = F.log_softmax(output[:, -1, :], dim=1).data.squeeze(0)
                best_scores, indices = scores.topk(BEAM_WIDTH)
                for score, index in zip(best_scores, indices):
                    new_seq = out_seq + [index.item()]
                    new_seq_score = seq_score + score
                    new_hypes.append((new_seq, new_seq_score))

            new_hypes_sorted = sorted(new_hypes, key=itemgetter(1), reverse=True)
            hypes = new_hypes_sorted[:BEAM_WIDTH]
            if new_hypes_sorted[0][0][-1] == EOS_ID:
                for i in range(BEAM_WIDTH):
                    token_finalist.append(new_hypes_sorted[i])
                break

        return token_finalist

def frame_stacking(x):
    newlen = len(x) // 3
    stacked_x = x[0:newlen * 3].reshape(newlen, 40 * 3)
    return stacked_x

if __name__ == '__main__':

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BEAM_WIDTH = 6

    feature_format = 'npy'
    LMFB_DIM = 40
    BEAM_WIDTH = int(sys.argv[1])
    NUM_CLASSES = int(sys.argv[2])
    EOS_ID = int(sys.argv[3])
    script_file = sys.argv[4]
    model_file = sys.argv[5]

    NUM_ENC_LAYERS = 12
    NUM_DEC_LAYERS = 6
    NUM_HEADS = 4
    DIM_MODEL = 256
    
    model = Transformer(d_feature = LMFB_DIM * 3, vocab_size = NUM_CLASSES, d_model = DIM_MODEL, n_enc_layers = NUM_ENC_LAYERS, n_dec_layers = NUM_DEC_LAYERS, n_heads = NUM_HEADS, dropoutrate = 0.1).to(DEVICE)
    #model = Transformer(d_feature = LMFB_DIM * 3, vocab_size = NUM_CLASSES, d_model = 512, n_layers = 6, n_heads = 8, dropoutrate = 0.1).to(DEVICE)
    #model = Transformer(d_feature = LMFB_DIM * 3, vocab_size = NUM_CLASSES, d_model = 512, n_layers = 6, n_heads = 8, dropoutrate = 0.0).to(DEVICE)
    model.eval()
    model.load_state_dict(torch.load(model_file))

    testing_data = [line for line in open(script_file)]
    for i in range(len(testing_data)):

        x_file = testing_data[i].strip()
        if len(x_file.split(' ', 1)) > 1:
            x_file, _ = x_file.split(' ', 1)
        if feature_format == 'htk':
            cpudat = load_dat(x_file)[:, :40]
        else:
            cpudat = np.load(x_file)

        #cpudat = frame_stacking(cpudat)

        xs = torch.from_numpy(cpudat).to(DEVICE).float().unsqueeze(0)

        prediction_beam = model(xs)

        print(x_file, end=" ")
        if len(prediction_beam) > 0:
            best_prediction = prediction_beam[0][0]
            for character in best_prediction:
                print(character, end=" ")
        print()
        sys.stdout.flush()
