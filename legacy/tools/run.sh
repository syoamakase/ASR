#!/bin/bash

/n/rd23/ueno/LM/local_bin/htk/bin/HCopy -T 1 -C config.lmfb.40ch.static -S scp.wav2lmfb

python feature_normalize.py -S scp.lmfb2lmfb
