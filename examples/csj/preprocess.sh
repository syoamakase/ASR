#!/bin/bash

# This script is based on run.sh in  https://github.com/kaldi-asr/kaldi/egs/csj/s5/run.sh

set -e # exit on error
mkdir -p tmp

vocab_type=bpe
vocab_size=7000
# hcopy_path=HCopy
# spm_path=

use_dev=false # Use the first 4k sentences from training data as dev set. (39 speakers.)

CSJDATATOP=/n/rd25/mimura/corpus/CSJ #/export/corpora5/CSJ/USB
#CSJDATATOP=/db/laputa1/data/processed/public/CSJ ## CSJ database top directory.
CSJVER=dvd  ## Set your CSJ format (dvd or usb).
           ## Usage    :
           ## Case DVD : We assume CSJ DVDs are copied in this directory with the names dvd1, dvd2,...,dvd17.
           ##            Neccesary directory is dvd3 - dvd17.
           ##            e.g. $ ls $CSJDATATOP(DVD) => 00README.txt dvd1 dvd2 ... dvd17
           ##
           ## Case USB : Neccesary directory is MORPH/SDB and WAV
           ##            e.g. $ ls $CSJDATATOP(USB) => 00README.txt DOC MORPH ... WAV fileList.csv
           ## Case merl :MERL setup. Neccesary directory is WAV and sdb

if [ ! -e data/csj-data/.done_make_all ]; then
echo "CSJ transcription file does not exist"
#local/csj_make_trans/csj_autorun.sh <RESOUCE_DIR> <MAKING_PLACE(no change)> || exit 1;
local/csj_make_trans/csj_autorun.sh $CSJDATATOP data/csj-data $CSJVER
fi
wait

[ ! -e data/csj-data/.done_make_all ]\
   && echo "Not finished processing CSJ data" && exit 1;

# Prepare Corpus of Spontaneous Japanese (CSJ) data.
# Processing CSJ data to KALDI format based on switchboard recipe.
# local/csj_data_prep.sh <SPEECH_and_TRANSCRIPTION_DATA_DIRECTORY> [ <mode_number> ]
# mode_number can be 0, 1, 2, 3 (0=default using "Academic lecture" and "other" data, 
#                                1=using "Academic lecture" data, 
#                                2=using All data except for "dialog" data, 3=using All data )
local/csj_data_prep.sh data/csj-data
for eval_num in eval1 eval2 eval3 ; do
   local/csj_eval_data_prep.sh data/csj-data/eval $eval_num
done

echo "Make acoustic features"
for dir_name in train eval1 eval2 eval3; do
    mkdir -p data/${dir_name}/wav.segment
    cut -d ' ' data/${dir_name}/wav.scp -f 1 | sed -e "s/^/mkdir -p data\/${dir_name}\/wav.segment\//g" | bash
    python tools/convert_kaldi2sox.py data/${dir_name}/segments data/${dir_name}/wav.scp data/${dir_name} > tmp/convert.sh
    bash tmp/convert.sh
    # make log mel-scale filter bank
    cut -d ' ' -f 3 tmp/convert.sh > data/${dir_name}/wavlist
    sed -e "s/wav\.segment/mel/g" -e "s/\.wav/\.htk/g" data/${dir_name}/wavlist > data/${dir_name}/mellist
    cut -d ' ' data/${dir_name}/wav.scp -f 1 | sed -e "s/^/mkdir -p data\/${dir_name}\/mel\//g" | bash
    paste -d ' ' data/${dir_name}/wavlist data/${dir_name}/mellist > tmp/scp.wav2mel
    HCopy -C config/config.lmfb.40ch.static -S tmp/scp.wav2mel
done

echo "Feature normalize"
python tools/feature_normalize.py -S data/train/mellist --ext htk --save_dir data/train/

echo "----- End preparing acoustic features ----"
echo "Start preparing script file"
python tools/clean_text.py -S data/train/text > data/train/input.txt
cut -d '|' -f 1- data/train/input.txt > data/train/input_sentencepice.txt
mkdir -p data/train/sentencepice/${vocab_type}_${vocab_size}/
spm_train --input=data/train/input_sentencepice.txt --model_prefix=data/train/sentencepice/${vocab_type}_${vocab_size}/model --vocab_size=${vocab_size} --character_coverage=1.0 --model_type=${vocab_type} --input_sentence_size=100000000 --bos_id=2 --eos_id=1 --unk_id=0

python tools/get_abspath.py -S data/train/input_sentencepice.txt > train_script.txt
python tools/sort_by_xlen.py -S train_script.txt > train_script.sort_xlen.txt
cat data/eval1/mellist data/eval2/mellist data/eval3/mellist > test_eval1_eval2_eval3.txt

echo "Complete!"
echo "The training data is train_script.sort_xlen.txt"
echo "The test data is test_eval1_eval2_eval3.txt"
echo "The spm_model is data/train/sentencepice/${vocab_type}_${vocab_size}/model.model"
