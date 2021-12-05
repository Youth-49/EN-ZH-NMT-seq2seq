#!/bin/bash

# if [ "$1" = "train" ]; then
# 	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./en_es_data/en-zh/train.en --train-tgt=./en_es_data/en-zh/train.zh --dev-src=./en_es_data/en-zh/dev.en --dev-tgt=./en_es_data/en-zh/dev.zh --vocab=vocab.json --cuda
# elif [ "$1" = "test" ]; then
#         CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./en_es_data/en-zh/test.en ./en_es_data/en-zh/test.zh outputs/test_outputs.txt --cuda
# elif [ "$1" = "vocab" ]; then
# 	python vocab.py --train-src=./en_es_data/en-zh/train.en --train-tgt=./en_es_data/en-zh/train.zh vocab.json
# else
# 	echo "Invalid Option Selected"
# fi

if [ "$1" = "train" ]; then
	python run.py --train=True --cuda=True
elif [ "$1" = "test" ]; then
        python run.py --decode=True
elif [ "$1" = "vocab" ]; then
	python vocab.py
else
	echo "Invalid Option Selected"
fi
