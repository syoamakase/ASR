#-*- coding:utf-8 -*-
import argparse
import os
import numpy as np
from struct import unpack

def remove_short_utt(script_filename):
    with open(script_filename) as f:
        for line in f:
            frame_length = int(line.strip().split('|')[2])
            if frame_length > 10:
                print(line.strip())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--script_filename', required=True)
    args = parser.parse_args()
    script_filename = args.script_filename

    remove_short_utt(script_filename)
