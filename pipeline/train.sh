#!/bin/sh

data_file="/home/mengqu2/projects/bio/sampled_pubmed.sen.phrase"
vector_file="/home/mengqu2/projects/bio/vec.emb"
model_file="model.txt"
seed_pos_file="seed-pos.txt"
seed_neg_file="seed-neg.txt"

echo "Extracting data..."
shuf ${data_file} > raw.shuf
./extract-data -data raw.shuf -seed ${seed_pos_file} -output data.pos -count cnt.pos -window 5 -label 1 -max 300
./extract-data -data raw.shuf -seed ${seed_neg_file} -output data.neg -count cnt.neg -window 5 -label 0 -max 300 
cat data.pos data.neg > data.all

echo "Training..."
./neu -data data.all -vector ${vector_file} -model ${model_file} -window 5 -mode 1