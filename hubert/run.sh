#!/bin/bash
# Get the start step from the first input argument, default is 1 if not provided
set -e
start_step=${1:-1}
echo "doing steps from step ${start_step}"

tsv_dir="YOUR_PATH_TO_LRS3(Processed)/433h_data/"
cluster_name="cluster_label_large"
fairseq_kmeans_path="YOUR_PATH_TO_FAIRSEQ/fairseq/examples/hubert/simple_kmeans"
splits=('train' 'valid' 'test')
if [ "$start_step" -le 1 ]; then
    for split in ${splits[@]}; do
        python avhubert_tsv2hubert.py $tsv_dir --input_split_tsv ${split}.tsv --output_split_tsv ${split}_${cluster_name}.tsv
    done
    echo "tsv generated"
fi
ckpt_path=YOUR_PATH_TO_HUBERT_CKPT
km_path=$tsv_dir/YOUR_KMEANS_MODEL.kmmodel
n_cluster=2000
layer=12
nshard=1
rank=0
feat_dir=$tsv_dir
lab_dir=$tsv_dir

cd $fairseq_kmeans_path
if [ "$start_step" -le 2 ]; then
    echo "dumping feature"
    for split in ${splits[@]}; do
        python dump_hubert_feature.py ${tsv_dir} ${split}_${cluster_name} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}
    done
fi

if [ "$start_step" -le 3 ]; then
    echo "doing clustering"
    python learn_kmeans.py ${feat_dir} train_${cluster_name} ${nshard} ${km_path} ${n_cluster} --percent 0.1
fi

if [ "$start_step" -le 4 ]; then
    echo "dumping label"
    for split in ${splits[@]}; do
        python dump_km_label.py ${feat_dir} ${split}_${cluster_name} ${km_path} ${nshard} ${rank} ${lab_dir}
    done
fi

if [ "$start_step" -le 5 ]; then
    echo "concatenating shards"
    for split in ${splits[@]}; do
        for rank in $(seq 0 $((nshard - 1))); do
        cat $lab_dir/${split}_${cluster_name}_${rank}_${nshard}.km
        done > $lab_dir/${split}_${cluster_name}.km
    done
fi

echo "all done"