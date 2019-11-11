import argparse
import copy
import numpy as np


def editDistance(r, h):
    '''
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0: 
                d[0][j] = j
            elif j == 0: 
                d[i][0] = i
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, deletion)
    return d


def getStepList(r, h, d):
    '''
    This function is to get the list of steps in the process of dynamic programming.
    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calulating the editting distance of h and r.
    '''
    x = len(r)
    y = len(h)
    results_stats = np.zeros((5)) # 'H', 'D', 'S', 'I', 'N'
    while True:
        if x == 0 and y == 0: 
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1] and r[x-1] == h[y-1]: 
            #results_list.append("h")
            results_stats[0] += 1
            x = x - 1
            y = y - 1
        elif x >= 1 and d[x][y] == d[x-1][y]+1:
            #results_list.append("d")
            results_stats[1] += 1
            x = x - 1
            y = y
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1]+1:
            #results_list.append("s")
            results_stats[2] += 1
            x = x - 1
            y = y - 1
        elif y >= 1 and d[x][y] == d[x][y-1]+1:
            results_stats[3] += 1
            #results_list.append("i")
            x = x
            y = y - 1
        else:
            print('There are some bugs')
    results_stats[4] = len(r)
    return results_stats


def wer(r, h):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split()) 
    """
    # build the matrix
    d = editDistance(r, h)

    # find out the manipulation steps
    results_list = getStepList(r, h, d)

    # print the result in aligned way
    result = float(d[len(r)][len(h)]) / len(r) * 100
    #print('WER {0:.2f}'.format(result))
    return results_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hyp_file')    
    parser.add_argument('ref_file')
    parser.add_argument('-w', '--word_list')
    parser.add_argument('--ignore_sos_eos', action='store_true')
    args = parser.parse_args()

    hyp_file = args.hyp_file
    ref_file = args.ref_file
    word_list = args.word_list
    ignore_sos_eos = args.ignore_sos_eos

    hyp_dict = {}
    ref_dict = {}

    word_dict = {}
    if word_list is not None:
        with open(word_list) as f:
            for line in f:
                word, word_id = line.strip().split(' ')
                word_dict[word_id] = word
        word_dict['<dummy>'] = '<dummy>'
    
    with open(hyp_file) as f:
        for line in f:
            # When the decoding didn't output anything
            if len(line.strip().split(' ')) == 1:
                file_id = line.strip()
                words = '<dummy>'
            else:
                file_id, words = line.strip().split(' ', 1)
            words = words.split(' ')
            
            # When word dict exists
            if len(word_dict) != 0:
                new_words = []
                for w in words:
                    new_words.append(word_dict[w])
                words = copy.deepcopy(new_words)

            if ignore_sos_eos and '<sos>' in words:
                words.remove('<sos>')
            if ignore_sos_eos and '<eos>' in words:
                words.remove('<eos>')

            hyp_dict[file_id] = words

    with open(ref_file) as f:
        for line in f:
            file_id, words = line.strip().split(' ', 1)
            words = words.split(' ')
            if ignore_sos_eos and '<sos>' in words and '<eos>' in words:
                words.remove('<sos>')
                words.remove('<eos>')
            ref_dict[file_id] = words

    results_all = np.zeros(5)
    for k, v in hyp_dict.items():
        assert (k in ref_dict.keys()), '{} is not found in ref'.format(k)

        ref_v = ref_dict[k]
        #print(k, end=' ')
        results_all += wer(ref_v, v)
    
    results_all = results_all.astype(np.int32)
    print('WER {0:.2f}% [H={1:d}, D={2:d}, S={3:d}, I={4:d}, N={5:d}]'.format(results_all[1:-1].sum()/ results_all[-1] * 100, results_all[0], results_all[1], results_all[2], results_all[3], results_all[4]))
    
