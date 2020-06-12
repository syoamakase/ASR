#-*- coding:utf-8 -*-
import argparse
import os
import numpy as np
from struct import unpack

def clean_text(script_filename):
    with open(script_filename) as f:
        for line in f:
            line = line.replace('  ', ' ')
            if len(line.strip().split(' ')) != 1:
                sp_line = line.strip().split(' ')
                print(sp_line[0], end='|')
                for words in sp_line[1:]:
                    if words != '' and words != '<sp>':
                        print(words.split('+')[0], end=' ')
            print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--script_filename', required=True)
    args = parser.parse_args()
    script_filename = args.script_filename

    clean_text(script_filename)
