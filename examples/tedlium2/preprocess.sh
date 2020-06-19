#!/bin/bash
vocab_type=bpe
vocab_size=1000

export SPH2PIPEPATH=

# download the data
local/download_data.sh

echo "Data preparation"
local/prepare_data.sh
for dset in dev test train; do
  utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}.orig data/${dset}
done

mkdir -p tmp
echo "Make acoustic features"
for dir_name in train dev test; do
    mkdir -p data/${dir_name}/wav.segment
    mkdir -p db/TEDLIUM_release2/${dir_name}/wav
    cut -d ' ' data/${dir_name}/wav.scp -f 6 | sed -e "s/sph/wav/g" > data/${dir_name}/wav.original.list
    #cut -d ' ' data/${dir_name}/wav.scp -f 2-6 | sed -e "s/^/${sph2pipe_path//\//\\/}/g" | paste -d ' ' - data/${dir_name}/wav.original.list > tmp/convert_sph2wav.sh
    cut -d ' ' data/${dir_name}/wav.scp -f 2-6 | paste -d ' ' - data/${dir_name}/wav.original.list > tmp/convert_sph2wav.sh
    bash tmp/convert_sph2wav.sh
    cut -d ' ' data/${dir_name}/wav.scp -f 1 | sed -e "s/^/mkdir -p data\/${dir_name}\/wav.segment\//g" | bash
    python tools/convert_kaldi2sox.py data/${dir_name}/segments data/${dir_name}/wav.scp data/${dir_name} > tmp/convert.sh
    bash tmp/convert.sh
    # make log mel-scale filter bank
    cut -d ' ' -f 3 tmp/convert.sh > data/${dir_name}/wavlist
    sed -e "s/wav\.segment/mel/g" -e "s/\.wav/\.htk/g" data/${dir_name}/wavlist > data/${dir_name}/mellist
    cut -d ' ' data/${dir_name}/wav.scp -f 1 | sed -e "s/^/mkdir -p data\/${dir_name}\/mel\//g" | bash
    paste -d ' ' data/${dir_name}/wavlist data/${dir_name}/mellist > tmp/scp.wav2mel
    HCopy -C config/config.lmfb.80ch.static -S tmp/scp.wav2mel
done

echo "Feature normalize"
python tools/feature_normalize.py -S data/train/mellist --ext htk --save_dir data/train/

echo "----- End preparing acoustic features ----"
# prepare language model dataset
mkdir -p data/lang/
mkdir -p data/train/sentencepice/${vocab_type}_${vocab_size}
gunzip -c db/TEDLIUM_release2/LM/*.en.gz > data/lang/train.txt
python tools/cut_longsentence.py data/lang/train.txt --threshold 1000 > data/lang/train_filt.txt

cut -d ' ' -f 2- data/train/text > data/train/input_sentencepiece.txt
spm_train --input=data/train/input_sentencepiece.txt --model_prefix=data/train/sentencepice/${vocab_type}_${vocab_size}/model --vocab_size=${vocab_size} --character_coverage=1.0 --model_type=${vocab_type} --input_sentence_size=100000000 --bos_id=2 --eos_id=1 --unk_id=0
cat data/train/input_sentencepiece.txt data/lang/train_filt.txt > train_lm.txt

python tools/clean_text.py -S data/train/text > data/train/input.txt
python tools/get_abspath.py -S data/train/input.txt > train_script.txt
python tools/sort_by_xlen.py -S train_script.txt -m 80 > tmp/train_script.sort_xlen.txt
python tools/remove_short_utt.py -S tmp/train_script.sort_xlen.txt > train_script.sort_xlen.txt
cat data/test/mellist data/dev/mellist | python3 -c "import sys; import os; [print(os.path.abspath(i.strip())) for i in sys.stdin]" > test_dev.txt

echo "Complete!" 
echo "The training data is train_script.sort_xlen.txt" > memo
echo "The test data is test_dev.txt" >> memo
echo "The trainign data for language model is train_lm.txt" >> memo
echo "The spm_model is data/train/sentencepice/${vocab_type}_${vocab_size}/model.model" >> memo
