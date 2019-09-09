# Speech Command Data Set

This repository is data directory for Speech Command Data set.
This data sets provided 2 versions (0.01 and 0.02).

In this repository, we prepare acoustic features (log mel-scale filter bank feattures) for an end-to-end speech recongnition.

## Preparation

1. We need to install python package

`pip install -r requirements.txt`

2. cd the directory which you like

`cd 0.01 #or 0.02`

3. execute path.sh

`./path.sh`

4. execute preprocess.sh

`./preprocess.sh`

It takes about 20 minutes to finish preprocess.sh.
After preprocessing, you can get word- and char-unit training scripts (`word/training_lmfb_list_word_id_random.txt` and `char/training_lmfb_list_char_id_random.txt`).

(5. clean files)

If you want to delete all files which you downloaded and generated, do clean.sh.

`./clean.sh`

## training

Back to ASR/ and change `train_script` and `num_classes` in hparams.py
You must change `num_classes` by the unit and version.

|   |#word|#char|
|:--:|:--:|:--:|
|0.01| 32 | 26 |
|0.02| 27 | 25 |


```
train_script = 'egs/speech_commands/word/training_lmfb_list_word_id_random.txt'

num_classes = 37 # or 32 or 26 or 25 
```



