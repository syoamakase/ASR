#!/bin/bash

find ./feature/wav/* -name '*.wav' | grep -v _background_noise > wav_list.txt

