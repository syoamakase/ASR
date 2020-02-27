# -*- coding: utf-8 -*-

import argparse
import sys
import os
import numpy as np
import matplotlib

isexists_display = os.getenv('DISPLAY')
if not isexists_display:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

MEAN_COUNT = 1

# def extract_loss(f, target='mtl'):
#     results = []
#     idx = 0
#     loss = 0
    
#     for line in f:
#         idx += 1
#         if ("loss =" in line) and (idx % 1 == 0):
#             if "loss_att" in line and target == 'attention':
#                 loss += float(line.replace("loss_att =  ", ""))
#                 if idx % MEAN_COUNT == 0:
#                     results.append(loss / MEAN_COUNT)
#                     loss = 0
#             elif "loss_ctc" in line and target == 'ctc':
#                 loss += float(line.replace("loss_ctc =  ", ""))
#                 if idx % MEAN_COUNT == 0:
#                     results.append(loss / MEAN_COUNT)
#                     loss = 0
#             elif target != 'attention' and target != 'ctc':
#                 loss += float(line.replace("loss = ", ""))
#                 if idx % MEAN_COUNT == 0:
#                     results.append(loss / MEAN_COUNT)
#                     loss = 0

    # return np.array(results)

def isfloat(string):
    try:
        float(string)  # cast string to float
        return True
    except ValueError:
        return False

def extract_att_loss(f, extract_char='loss'):
    results = []
    idx = 0
    loss = 0
    
    for line in f:
        idx += 1
        #if idx < 30000:
        #    continue
        if ("{} =".format(extract_char) in line) and (idx % 1 == 0):
            line = line.strip()
            if not isfloat(line.replace("{} = ".format(extract_char), "")):
                #print(line.replace("loss = ", ""))
                continue
            loss += float(line.replace("{} = ".format(extract_char), ""))
            if idx % MEAN_COUNT == 0:
                results.append(loss / MEAN_COUNT)
                loss = 0
        if len(results) > 300000:
            break 
    return np.array(results)

if __name__ == "__main__":
    argc = len(sys.argv)
    extract_char = []
    for i in reversed(range(1, argc)):
        if not os.path.exists(sys.argv[i]):
            argc -= 1
            extract_char.append(sys.argv[i].strip())
            print(sys.argv[i])
        else:
            break
            #extract_char.append('loss')

    if len(extract_char) == 0:
        extract_char = ['loss']

    if argc == 1:
        print("usage: python {} <log file1> ...".format(sys.argv[0]))
        sys.exit(1)

    for i in range(1, argc):
        filename = sys.argv[i]
        for ext_char in extract_char:
            with open(filename, 'r') as f:
                losses = extract_att_loss(f, ext_char)
                plt.plot(losses, label=filename+'_'+ext_char, alpha=0.6)
        #plt.hold(True)

    #plt.ylim(0, 200)
    plt.legend()
    if not isexists_display:
        plt.savefig('results.png')
    else:
        plt.show()
