#!/bin/bash
set -e

tsv_dir="/home/yfliu/datasets/lrs3/30h_data/"
config_path='/home/yfliu/SpeechTokenizer/ckpt/speechtokenizer_hubert_avg_config.json'
ckpt_path='/home/yfliu/SpeechTokenizer/ckpt/SpeechTokenizer.pt'
lab_dir=$tsv_dir
splits=('test' 'valid' 'train')
token_name="hubert_avg_config"
nshard=1
rank=0
min_seconds=10

for split in ${splits[@]}; do
    python ../hubert/avhubert_tsv2hubert.py $tsv_dir --input_split_tsv ${split}.tsv --output_split_tsv ${split}_${token_name}.tsv
done
echo "tsv generated"

echo "dumping label"
for split in ${splits[@]}; do
    python dump_labels.py  ${tsv_dir} ${split}_${token_name} ${ckpt_path} ${config_path} ${nshard} ${rank} ${lab_dir} ${min_seconds}
done

echo "concatenating shards"
for split in ${splits[@]}; do
    for rank in $(seq 0 $((nshard - 1))); do
    cat $lab_dir/${split}_${token_name}_${rank}_${nshard}.st
    done > $lab_dir/${split}_${token_name}.st
done