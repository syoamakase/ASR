import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--list_file')
    parser.add_argument('-v', '--vocab_file')
    args = parser.parse_args()

    list_file = args.list_file
    vocab_file = args.vocab_file

    vocab_dict = {}
    with open(vocab_file) as f:
        for line in f:
            word, word_id = line.strip().split(' ')
            vocab_dict[word] = word_id

    with open(list_file) as f:
        for line in f:
            sp_line = line.strip().split(' ')
            print(sp_line[0], end=' ')
            for word in sp_line[1:]:
                print(vocab_dict[word], end=' ')
            print()
