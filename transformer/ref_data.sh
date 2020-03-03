#!/bin/bash
#SBATCH --gres=gpu:RTX:2
#SBATCH --partition=sacs

python train-transformer-multi-cnn_dataset.py 1059 script.e2e.bpe.1k.sacs01.sorted.bucket120 model.multi.e12.d6.h4.d256.specaug.dataset > log.model.multi.e12.d6.h4.d256.specaug.dataset100-220
