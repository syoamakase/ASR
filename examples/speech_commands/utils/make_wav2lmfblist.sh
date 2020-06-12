#!/bin/bash

cat wav_list.txt | sed -e "s/\/wav\//\/lmfb\//g" -e "s/.wav/.npy/g" > lmfb_list.txt

paste wav_list.txt lmfb_list.txt -d " " > wav2lmfb_list.txt
