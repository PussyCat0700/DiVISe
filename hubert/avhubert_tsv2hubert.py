import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("avhubert_tsvdir")
    parser.add_argument("--input_split_tsv", default='train.tsv', help='Should only contain data used for training. e.g. train.tsv')
    parser.add_argument("--output_split_tsv", default='kmeans.tsv', help='e.g. kmeans.tsv')
    args = parser.parse_args()
    input_tsv = os.path.join(args.avhubert_tsvdir, args.input_split_tsv)
    output_tsv = os.path.join(args.avhubert_tsvdir, args.output_split_tsv)
    with open(input_tsv, 'r') as f:
        all_lines = f.readlines()
    all_lines = all_lines[1:]
    all_lines = [x.split()[-3].strip()+'\t'+x.split()[-1].strip()+'\n' for x in all_lines]
    with open(output_tsv, 'w') as f:
        f.write("/\n")
        f.writelines(all_lines)