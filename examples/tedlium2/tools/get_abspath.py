import argparse
import os

parser = argparse.ArgumentParser() 
parser.add_argument('-S', '--script_filename', required=True)
parser.add_argument('--ext', default='htk')
args = parser.parse_args()
script_filename = args.script_filename
ext = args.ext

with open(script_filename) as f:
    for line in f:
        file_id, text = line.strip().split('|',1)
        speaker_id = file_id.split('-')[0]
        file_name = f'data/train/mel/{speaker_id}/{file_id}.{ext}'
        print(f'{os.path.abspath(file_name)}|{text}')
