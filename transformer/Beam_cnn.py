import torch

import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0).to(DEVICE)
    return np_mask


def init_vars(src, src_dummy, model, k=10, init_tok=0):

    src_pad = 0 #19147
    #init_tok = 0 #TRG.vocab.stoi['<sos>']
    max_len = 200
    # src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    src_mask = (src_dummy != src_pad).unsqueeze(-2)

    src, src_mask = model.cnn_encoder(src, src_mask)
    e_output = model.encoder(src, src_mask)
    #import pdb; pdb.set_trace()

    outputs = torch.LongTensor([[init_tok]])
    outputs = outputs.to(DEVICE)

    trg_mask = nopeak_mask(1)

    out = model.out(model.decoder(outputs, e_output, src_mask, trg_mask))
    out = F.softmax(out, dim=-1)

    probs, ix = out[:, -1].data.topk(k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)

    outputs = torch.zeros(k, max_len).long()
    outputs = outputs.to(DEVICE)
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]

    e_outputs = torch.zeros(k, e_output.size(-2),e_output.size(-1))
    e_outputs = e_outputs.to(DEVICE)
    e_outputs[:, :] = e_output[0]

    return outputs, e_outputs, log_scores, src_mask

def k_best_outputs(outputs, out, log_scores, i, k):

    probs, ix = out[:, -1].data.topk(k)

    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
    k_probs, k_ix = log_probs.view(-1).topk(k)

    # import pdb; pdb.set_trace()
    # row = k_ix // k
    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)

    return outputs, log_scores

def beam_search(src, src_dummy, model, init_tok=0, eos_tok=1):

    src_pad = 0
    max_len = 200
    k = 20
    outputs, e_outputs, log_scores, src_mask = init_vars(src, src_dummy, model, k, init_tok)
    #eos_tok = 1 #TRG.vocab.stoi['<eos>']
    # src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    #src_mask = (src_dummy != src_pad).unsqueeze(-2)

    ind = None
    lengths = None
    for i in range(2, max_len):

        trg_mask = nopeak_mask(i)

        out = model.out(model.decoder(outputs[:, :i], e_outputs, src_mask, trg_mask))

        out = F.softmax(out, dim=-1)
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, k)

        #if (outputs==eos_tok).nonzero().size(0) == k:
        #    alpha = 0.7
        #    div = 1/((outputs==eos_tok).nonzero()[:,1].type_as(log_scores)**alpha)
        #    _, ind = torch.max(log_scores * div, 1)
        #    ind = ind.data[0]
        #    break
        if outputs[0][i] == eos_tok:
            lengths = i
            break

   # if ind is None:
   #     #length = (outputs[0]==eos_tok).nonzero()[0]
   #     results = ''
   #     prev_label = [0, 1]
   #     for tok in outputs[0]:
   #         if tok == eos_tok:
   #             results += str(tok.item())
   #             break
   #         else:
   #             if prev_label[0] == tok.item():
   #                 prev_label[0] = tok.item()
   #                 prev_label[1] += 1
   #             else:
   #                 prev_label[0] = tok.item()
   #                 prev_label[1] = 1

   #             if prev_label[1] == 3:
   #                 break
   #             results += str(tok.item()) + ' '
   #         
   #     return results
   #     #return ' '.join([tok for tok in outputs[0][1:length]])

   #     # return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:length]])

   # else:
   #     #length = (outputs[ind]==eos_tok).nonzero()[0]
   #     #print("ind", outputs)
   #     results = ''
   #     prev_label = [0, 1]
   #     for tok in outputs[0]:
   #         if tok == eos_tok:
   #             results += str(tok.item())
   #             break
   #         else:
   #             if prev_label[0] == tok.item():
   #                 prev_label[0] = tok.item()
   #                 prev_label[1] += 1
   #             else:
   #                 prev_label[0] = tok.item()
   #                 prev_label[1] = 1

   #             if prev_label[1] == 3:
   #                 break
   #             results += str(tok.item()) + ' '
    if lengths is None:
        return ''
    return ' '.join([str(tok) for tok in outputs[0][:lengths+1].cpu().numpy()])
        #return ' '.join([tok for tok in outputs[ind][1:length]])

        # return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])
