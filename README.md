## Sequence-to-sequence speech recognition toolkit

This is sequence-to-sequence speech recognition toolkit.
This script doesn't include preprocess (segment wave files, tanscriptions, and word labels).

## TODO

- preprocess (CSJ, Librispeech)
- zoneout
- shallow fusion
- real-time version
- CTC (?)

## Requirements

Python >= 3.6.0  
PyTorch >= 1.0

We reccomend you to prepare [Anaconda 3](https://www.anaconda.com/distribution/).

## Quick Start

### Installation

`pip install -r requirements.txt`

### Train

`python train.py`

### Test

`python test.py`


## Notice

- When `debug_mode = 'visdom'` and you use a remote server, specify a visdom server's ip address in `viz = Visdom()` of train.py ex) `viz = Visdom('192.168.0.2')`

## For previous version developers

You can use previous version's model when you specify `legacy = True` in hparams.py.

## Reference

 [1] Jan Chorowski, Dzmitry Bahdanau, Dmitriy Serdyuk,Kyunghyun Cho, and Yoshua Bengio, “Attention-based models for speech recognition,” inAdvances in Neural InformationProcessing Systems (NIPS), 2015, pp.577–585.