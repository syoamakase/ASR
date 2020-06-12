# Speech Command Data Set

This repository is data directory for Speech Command Data set.
This data sets provided 2 versions (0.01 and 0.02).

In this repository, we prepare the acoustic features (log mel-scale filter bank features) for an end-to-end speech recongnition.

As a results, you can get `word/training_lmfb_list_word.txt`, `char/training_lmfb_list_char_id_random.txt`, `word/training_wav_list_word.txt`, `char/training_wav_list_char_id_random.txt`.

## Preparation

1. Install python package

`pip install -r requirements.txt`

2. cd the directory which you like

`cd 0.01 #or 0.02`

3. Execute preprocess.sh

`./preprocess.sh`

It takes about 20 minutes to finish preprocess.sh.  
After preprocessing, you can get word- and char-unit training scripts (`word/training_lmfb_list_word_id_random.txt` and `char/training_lmfb_list_char_id_random.txt`).

(4. clean files)

If you want to delete all files which you downloaded and generated, do clean.sh.

`./clean.sh`

## Training

Back to ASR/ and change `train_script` and `num_classes` in hparams.py
You must change `num_classes` by the unit and version.

|   |#word|#char|
|:--:|:--:|:--:|
|0.01| 32 | 25 |
|0.02| 37 | 26 |

```
train_script = 'egs/speech_commands/0.01/word/training_lmfb_list_word_id_random.txt'

num_classes = 37 # or 32 or 26 or 25 
```

## Evaluation

You set `test_script=egs/speech_commands/0.01/word/validation_lmfb_list_word.txt` and do `test.py`.

`python test.py --load_name checkpoint/network.epoch20 > e20.val.txt`

testing or validation list like "egs/speech_commands/0.01/word/validation_lmfb_list_word.txt" includes the correct words.
Don't worry!!  
In test.py, we don't use the correct words.

After finishing test.py, we calculate [word error rate](https://en.wikipedia.org/wiki/Word_error_rate).

`python tools/calc_wer.py --ignore_sos_eos e20.val.txt egs/speech_commands/0.01/word/validation_lmfb_list_word.txt`
