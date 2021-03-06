**Don't use pytorch==1.4.0!!!!!!**

This is sequence-to-sequence speech recognition toolkit.

![stars.svg](https://img.shields.io/github/stars/syoamakase/ASR)

## Requirements

Python >= 3.7.0  
PyTorch >= 1.2.0  

We highly recommend you to prepare [Anaconda 3](https://www.anaconda.com/distribution/).

## Installation

For preprocess, we need [sentencepice](https://github.com/google/sentencepiece) and [HTK](http://htk.eng.cam.ac.uk/download.shtml)

`pip install -r requirements.txt`

## Preprocess

`examples/*/preprocess.sh` is a preprocess script.
After `preprocess.sh`, you can get the training data and test data.

### Train

`python train.py --hp_file hparams.py`

While you do a training, you can check the loss curve using tensorboard.
When you set the specific gpu(s), please set such as `CUDA_VISIBLE_DEVICES=0`
After `tensorboard --logdir <log dir>` and accessing `localhost:6006` on your web browser, you can check.

### Test

`python test.py --load_name <model name> --hp_file <hparams.py path>`

If you don't specify `--hp_file`, `test.py` searches the directory of <model name>

## Results WER[%]

## CSJ

|                    |eval 1 |eval 2 |eval 3 |
|--------------------|------:|------:|------:|
|CSJ-APS+SPS (7k BPE)|8.86   |8.21   |6.28   |

## LibriSpeech

|             |dev clean |dev other |test clean |test other |
|-------------|---------:|---------:|----------:|----------:|
|960h (word)  |6.23      |14.41     |6.29       |14.94      |
|960h (1k BPE)|4.05      |11.62     |4.19       |11.88      |


## Tedlium2 (1k BPE)

|                           |dev       | test     |
|---------------------------|---------:|---------:|
|Attention + 40 (flat start)|12.29     |10.44     |


## TODO

- More faster `tools/calc_wer.py` 
- preprocess (Librispeech)
- shallow fusion (including LM training)

## Reference paper

### We developed the attention model based on
[1] Jan Chorowski, Dzmitry Bahdanau, Dmitriy Serdyuk, Kyunghyun Cho, and Yoshua Bengio, “Attention-based models for speech recognition,” in Advances in Neural InformationProcessing Systems (NIPS), 2015, pp.577–585.

### We applied the label smoothing to improve ASR performance
[2] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna, "Rethinking the inception architecture for computer vision" in IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp.2818-2826.

### We also used SpecAugment
[3] Daniel S. Park, William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D. Cubuk, and Quoc V. Le, "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition" in Proc. Interspeech, 2019, pp.2613--2617.

## Reference link

[https://github.com/espnet/espnet](https://github.com/espnet/espnet)
