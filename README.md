## Sequence-to-sequence speech recognition toolkit

This is sequence-to-sequence speech recognition toolkit.
This script doesn't include preprocess (segment wave files, tanscriptions, and word labels).

## TODO

- preprocess
- zoneout
- shallow fusion
- real-time version

## Requirements

Python >= 3.6  
PyTorch >= 0.4

We reccomend you to prepare [Anaconda 3](https://www.anaconda.com/distribution/).

## Quick Start

### Train

`python train.py`

### Test

`python test.py`

## For previous version developers

You can use previous version's model when you specify `legacy = True` in hparams.py.

## Reference

