import argparse
import os

parser = argparse.ArgumentParser() 
parser.add_argument('script_file')
parser.add_argument('--ext', default='htk')
args = parser.parse_args()
script_file = args.script_file 
ext = args.ext

with open(script_file) as f:
    for line in f:
        file_id, text = line.split('|',1)
        speaker_id = file_id.split('_')[0]
        file_name = f'data/train/mel/{speaker_id}/{file_id}.{ext}'
        print(f'{os.path.abspath(file_name)}|text')