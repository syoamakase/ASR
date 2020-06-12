import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('list_txt')
    parser.add_argument('--mode', help='char or word')

    args = parser.parse_args()
    list_txt = args.list_txt
    mode = args.mode

    with open(list_txt) as f:
        for line in f:
            sp_line = line.strip().split('/')
            word = sp_line[-2]
            print(os.path.abspath(line.strip()) + ' <sos>', end=' ')
            if mode == 'word':
                print(word, end=' ')
            elif mode == 'char':
                for char in word:
                    print(char, end=' ')
            print('<eos>')


