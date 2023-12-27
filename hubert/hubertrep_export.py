
import logging
import os
import sys
import numpy as np
import argparse

from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

def export_hubert_feature(feat_dir, kmeans_split, tsv_split):
    feat_path = f"{feat_dir}/{kmeans_split}.npy"
    leng_path = f"{feat_dir}/{kmeans_split}.len"
    tsv_path = f"{feat_dir}/{tsv_split}.tsv"
    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()
    feat = np.load(feat_path, mmap_mode="r")
    with open(tsv_path, "r") as f:
        root = f.readline().rstrip()
        wav_paths = [line.rstrip().split("\t")[0] for line in f]
        pbar = tqdm(zip(offsets, lengs, wav_paths), total=len(wav_paths))
    for offset, leng, wav_path in pbar:
        dir_path = os.path.dirname(wav_path)
        file_prefix = os.path.basename(wav_path).split('.')[0]
        save_path = os.path.join(dir_path, f"{file_prefix}_{kmeans_split}.npy")
        feat_item = feat[offset: offset + leng]
        np.save(save_path, feat_item)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("avhubert_tsvdir")
    parser.add_argument("--tsv_split", default='cluster_label', help='(replace with yours) output of avhubert_tsv2hubert.py. e.g. kmeans refers to kmeans.tsv')
    parser.add_argument("--kmeans_split", default='cluster_label', help='(replace with yours) e.g. kmeans refers to kmeans.npy and kmeans.len')
    args = parser.parse_args()
    export_hubert_feature(args.avhubert_tsvdir, args.tsv_split, args.kmeans_split)