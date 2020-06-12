This is sequence-to-sequence speech recognition toolkit.
This script doesn't include preprocess (segment wave files, tanscriptions, and word labels).

## Requirements

Python >= 3.7.0
PyTorch >= 1.2.0

We highly recommend you to prepare [Anaconda 3](https://www.anaconda.com/distribution/).

## Installation

For preprocess, we need [sentencepice](https://github.com/google/sentencepiece) and [HTK](http://htk.eng.cam.ac.uk/download.shtml)

`pip install -r requirements.txt`

## Preprocess

`examples` directory is available corpus

### Train

`python train.py --hp_file hparams.py`

### Test

`python test.py --load_name <model name>`

## Results

## CSJ

Table 1 shows word error rate (WER[%]) on CSJ corpus.  
First rows mean training corpus and first columns mean test set.
We trained 40 epochs and chose minimum WER from 40 epochs model.

|            |units |#vocab |CSJ-APS|CSJ-SPS|
|------------|-----:|------:|------:|------:|
|CSJ-APS     |word  |19146  |10.68  |17.38  |
|CSJ-SPS     |word  |24826  |21.97  |8.88   |
|CSJ-APS+SPS |word  |34331  |9.56   |8.57   |
|CSJ-APS+SPS |BPE-500|6027  |8.35   |6.64   |

## LibriSpeech

|             |dev clean |dev other |test clean |test other |
|-------------|---------:|---------:|----------:|----------:|
|100h         |xx.xx     |xx.xx     |xx.xx      |xx.xx      |
|960h(word)   |6.23      |14.41     |6.29       |14.94      |
|960h(1k BPE) |4.05      |11.62     |4.19       |11.88      |


## Tedlium2 (1k BPE)

|                           |dev       | test     |
|---------------------------|---------:|---------:|
|Attention + 40 (flat start)|12.88     |10.84     |


## TODO

- More faster `tools/calc_wer.py` 
- preprocess (Tedlium2, Librispeech)
- shallow fusion

## Reference

### We developed the attention model based on
[1] Jan Chorowski, Dzmitry Bahdanau, Dmitriy Serdyuk, Kyunghyun Cho, and Yoshua Bengio, “Attention-based models for speech recognition,” in Advances in Neural InformationProcessing Systems (NIPS), 2015, pp.577–585.

### We applied the label smoothing to improve ASR performance
[2] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna, "Rethinking the inception architecture for computer vision" in IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp.2818-2826.

### We also used SpecAugment
[3] Daniel S. Park, William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D. Cubuk, and Quoc V. Le, "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition" in Proc. Interspeech, 2019, pp.2613--2617.
