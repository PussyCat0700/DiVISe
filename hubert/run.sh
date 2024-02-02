#!/bin/bash
# Get the start step from the first input argument, default is 1 if not provided
set -e
start_step=${1:-1}
echo "doing steps from step ${start_step}"
skip_clustering=false
if [ "$2" = "-simple" ]; then
    echo "clustering will be skipped"
    skip_clustering=true
fi

tsv_dir="/home/yfliu/datasets/lrs3/30h_data/"
cluster_name="cluster_label"
fairseq_kmeans_path="/home/yfliu/av_hubert/fairseq/examples/hubert/simple_kmeans"
splits=('train' 'valid' 'test')
if [ "$start_step" -le 1 ]; then
    for split in ${splits[@]}; do
        python avhubert_tsv2hubert.py $tsv_dir --input_split_tsv ${split}.tsv --output_split_tsv ${split}_${cluster_name}.tsv
    done
    echo "tsv generated"
fi
ckpt_path=/home/yfliu/hifi-gan/hubert/hubert_base_ls960.pt
km_path=/home/yfliu/datasets/lrs3/433h_data/ls960base.kmmodel
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

if [ "$start_step" -le 3 ] && ! $skip_clustering; then
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