import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('list_txt')
    args = parser.parse_args()

    list_txt = args.list_txt

    vocab = {}
    with open(list_txt) as f:
        for line in f:
            sp_line = line.strip().split(' ')[1:] 
            for word in sp_line:
                vocab[word] = 1

    sorted_word = sorted(vocab.keys())
    for i, x in enumerate(sorted_word):
        print(x, i)       
