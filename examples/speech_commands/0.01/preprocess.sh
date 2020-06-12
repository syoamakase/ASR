#!/bin/bash -x

ln -s ../utils utils

# make feature directory
mkdir -p feature/wav feature/lmfb common

wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz

tar -zxf speech_commands_v0.01.tar.gz -C ./feature/wav/

# make lmfb feature directory
find ./feature/wav/ -type d | grep -v "_background_noise_" | sed -e "s/wav/lmfb/g" -e "s/^/mkdir -p /g" | bash

# make wav_list.txt, lmfb_list.txt, and wav2lmfb_list.txt
./utils/make_wavlist.sh
./utils/make_wav2lmfblist.sh

# extract log mel-scale filter bank features from wav
# It takes 5~10 minutes.
python ./utils/lmfb-htklike_v1.py wav2lmfb_list.txt

# split training, validation, and testing data
cat prep/training_list.txt | sed -e "s/^/feature\/lmfb\//g" -e "s/$/.npy/g" | python utils/get_abspath.py > common/training_lmfb_list.txt
cat prep/testing_list.txt | sed -e "s/^/feature\/lmfb\//g" -e "s/$/.npy/g" | python utils/get_abspath.py > common/testing_lmfb_list.txt
cat prep/validation_list.txt | sed -e "s/^/feature\/lmfb\//g" -e "s/$/.npy/g"| python utils/get_abspath.py > common/validation_lmfb_list.txt

# for waveform input
cat prep/training_list.txt | sed -e "s/^/feature\/wav\//g" -e "s/$/.wav/g" | python utils/get_abspath.py > common/training_wav_list.txt
cat prep/testing_list.txt | sed -e "s/^/feature\/wav\//g" -e "s/$/.wav/g" | python utils/get_abspath.py > common/testing_wav_list.txt
cat prep/validation_list.txt | sed -e "s/^/feature\/wav\//g" -e "s/$/.wav/g"| python utils/get_abspath.py > common/validation_wav_list.txt

# normalize data N(0,1)
python utils/feature_normalize.py -S common/training_lmfb_list.txt
python utils/feature_normalize.py -S common/testing_lmfb_list.txt
python utils/feature_normalize.py -S common/validation_lmfb_list.txt

#
mkdir -p word char

# make vocabulary
python utils/make_labels.py common/training_lmfb_list.txt --mode char > char/training_lmfb_list_char.txt
python utils/make_labels.py common/testing_lmfb_list.txt --mode char > char/testing_lmfb_list_char.txt
python utils/make_labels.py common/validation_lmfb_list.txt --mode char > char/validation_lmfb_list_char.txt

python utils/make_labels.py common/training_lmfb_list.txt --mode word > word/training_lmfb_list_word.txt
python utils/make_labels.py common/testing_lmfb_list.txt --mode word > word/testing_lmfb_list_word.txt
python utils/make_labels.py common/validation_lmfb_list.txt --mode word > word/validation_lmfb_list_word.txt

python utils/make_labels.py common/training_wav_list.txt --mode char > char/training_wav_list_char.txt
python utils/make_labels.py common/testing_wav_list.txt --mode char > char/testing_wav_list_char.txt
python utils/make_labels.py common/validation_wav_list.txt --mode char > char/validation_wav_list_char.txt

python utils/make_labels.py common/training_wav_list.txt --mode word > word/training_wav_list_word.txt
python utils/make_labels.py common/testing_wav_list.txt --mode word > word/testing_wav_list_word.txt
python utils/make_labels.py common/validation_wav_list.txt --mode word > word/validation_wav_list_word.txt

python utils/make_vocab.py char/training_lmfb_list_char.txt > char/char_id.txt
python utils/make_vocab.py word/training_lmfb_list_word.txt > word/word_id.txt


python utils/make_id_file.py -l word/training_lmfb_list_word.txt -v word/word_id.txt > word/training_lmfb_list_word_id.txt
#python utils/make_id_file.py -l word/testing_lmfb_list_word.txt -v word/word_id.txt > word/testing_lmfb_list_word_id.txt
#python utils/make_id_file.py -l word/validation_lmfb_list_word.txt -v word/word_id.txt > word/validation_lmfb_list_word_id.txt
python utils/make_id_file.py -l word/training_wav_list_word.txt -v word/word_id.txt > word/training_wav_list_word_id.txt

python utils/make_id_file.py -l char/training_lmfb_list_char.txt -v char/char_id.txt > char/training_lmfb_list_char_id.txt
#python utils/make_id_file.py -l char/testing_lmfb_list_char.txt -v char/char_id.txt > char/testing_lmfb_list_char_id.txt
#python utils/make_id_file.py -l char/validation_lmfb_list_char.txt -v char/char_id.txt > char/validation_lmfb_list_char_id.txt
python utils/make_id_file.py -l char/training_wav_list_word.txt -v char/char_id.txt > char/training_wav_list_word_id.txt

sort -R word/training_lmfb_list_word_id.txt > word/training_lmfb_list_word_id_random.txt
sort -R char/training_lmfb_list_char_id.txt > char/training_lmfb_list_char_id_random.txt

sort -R word/training_wav_list_word_id.txt > word/training_wav_list_word_id_random.txt
sort -R char/training_wav_list_char_id.txt > char/training_wav_list_char_id_random.txt
