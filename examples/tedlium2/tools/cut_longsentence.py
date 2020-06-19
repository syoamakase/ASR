#-*- coding: utf-8 -*-
import argparse

def cut_longsentence(script_file, threshold):
    with open(script_file) as f:
        for i, line in enumerate(f):
            sp_line = line.strip().split(' ')
            if len(sp_line) < threshold:
                print(line.strip())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('script_file')
    parser.add_argument('--threshold', type=int, default=1000)
    args = parser.parse_args()
    script_file = args.script_file
    threshold = args.threshold

    cut_longsentence(script_file, threshold)

