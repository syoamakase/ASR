import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('segment')
parser.add_argument('wavscp')
parser.add_argument('csj_path')
args = parser.parse_args()
filename = args.segment
wavscp = args.wavscp
csj_path = os.path.abspath(args.csj_path)

file_dict = {}
with open(wavscp) as f:
    for line in f:
        sp_line = line.strip().split(' ')
        file_id = sp_line[0]
        wav_path = sp_line[2]
        file_dict[file_id] = wav_path 

with open(filename) as f:
    for line in f:
        sp_line = line.strip().split(' ')
        save_file = sp_line[0]
        file_id = sp_line[1]
        start = float(sp_line[2])
        end = float(sp_line[3])
        print(f'sox {file_dict[file_id]} {csj_path}/wav.segment/{file_id}/{save_file}.wav trim {start} ={end}')
