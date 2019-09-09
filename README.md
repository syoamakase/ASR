This is sequence-to-sequence speech recognition toolkit.
This script doesn't include preprocess (segment wave files, tanscriptions, and word labels).
## TODO

- preprocess (CSJ, Librispeech)
- zoneout
- shallow fusion
- real-time version

## Requirements

Python >= 3.6.0  
PyTorch >= 1.0

We recommend you to prepare [Anaconda 3](https://www.anaconda.com/distribution/).

## Quick Start

### Installation

`pip install -r requirements.txt`

### Train

`python train.py`

### Test

`python test.py`

## File Format

### Train

`<file name> <label sequence (int)>`  
file1.npy 1 2 3 4

### Test

`<file name>`
file1.npy

## Notice

- When `debug_mode = 'visdom'` and you use a remote server, specify a visdom server's ip address in `viz = Visdom()` of train.py  
  ex) `viz = Visdom('192.168.0.2')`


## Results

## CSJ

Table 1 Word error rate (WER[%]) on CSJ corpus.  
First rows mean training corpus and first columns mean test set.
We trained 40 epochs and chose minimum WER from 40 epochs model.
**BOLD** means latest results using this repository (others come from legacy model).

|            |#vocab |CSJ-APS|CSJ-SPS|
|------------|------:|------:|------:|
|CSJ-APS     |19146  |**11.91**|19.22  |
|CSJ-SPS     |24826  |23.30  |9.69   |
|CSJ-APS+SPS |34331  |10.30  |9.06   |

## LibriSpeech

|            |dev clean |dev other |test clean |test other |
|------------|---------:|---------:|----------:|----------:|
|100h        |xx.xx     |xx.xx     |xx.xx      |xx.xx      |


## For previous version developers

You can use previous version's model when you specify `legacy = True` in hparams.py.

## Reference

[1] Jan Chorowski, Dzmitry Bahdanau, Dmitriy Serdyuk, Kyunghyun Cho, and Yoshua Bengio, “Attention-based models for speech recognition,” in Advances in Neural InformationProcessing Systems (NIPS), 2015, pp.577–585.

