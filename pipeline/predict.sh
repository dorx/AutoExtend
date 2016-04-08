#!/bin/sh

data_file="/home/mengqu2/projects/bio/sampled_pubmed.sen.phrase"
vector_file="/home/mengqu2/projects/bio/vec.emb"
model_file="model.txt"
candidate_file="cand.txt"

./extract-data -data raw.shuf -seed ${candidate_file} -output data.test -count cnt.test -window 5 -label 0 -max 100

./neu -data data.test -vector ${vector_file} -output out.test -model ${model_file} -mode 2 -window 5

python rank-neu.py out.test ${candidate_file} cnt.test rank.txt
