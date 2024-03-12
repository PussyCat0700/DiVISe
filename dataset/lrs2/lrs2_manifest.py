# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

def main():
    import argparse
    parser = argparse.ArgumentParser(description='LRS3 tsv preparation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lrs2', type=str, help='lrs2 root dir')
    args = parser.parse_args()
    # see last step of lrs2_prepare.py for reference.
    sequence = ['train', 'val', 'test', 'short-pretrain',]
    label_lists = []
    args.lrs2 = os.path.join(args.lrs2, 'mvlrs_v1')
    file_list = f"{args.lrs2}/file.list"
    assert os.path.isfile(file_list) , f"{file_list} not exist -> run lrs2_prepare.py first"
    for split in sequence:
        label_list = f"{args.lrs2}/{split}_label.list"
        assert os.path.isfile(label_list) , f"{label_list} not exist -> run lrs2_prepare.py first"
        label_lists.append((split, label_list))
    labels = []
    for (split, label_list) in label_lists:
        labels += [(split, x.strip().lower()) for x in open(label_list).readlines()]
    nframes_audio_file, nframes_video_file = f"{args.lrs2}/nframes.audio", f"{args.lrs2}/nframes.video"
    assert os.path.isfile(nframes_audio_file) , f"{nframes_audio_file} not exist -> run count_frames.py first"
    assert os.path.isfile(nframes_video_file) , f"{nframes_video_file} not exist -> run count_frames.py first"

    audio_dir, video_dir = f"{args.lrs2}/audio", f"{args.lrs2}/video"

    def setup_target(target_dir, train, valid, test):
        for name, data in zip(['train', 'valid', 'test'], [train, valid, test]):
            with open(f"{target_dir}/{name}.tsv", 'w') as fo:
                fo.write('/\n')
                for fid, _, nf_audio, nf_video in data:
                    fo.write('\t'.join([fid, os.path.abspath(f"{video_dir}/{fid}.mp4"), os.path.abspath(f"{audio_dir}/{fid}.wav"), str(nf_video), str(nf_audio)])+'\n')
            with open(f"{target_dir}/{name}.wrd", 'w') as fo:
                for _, label, _, _ in data:
                    fo.write(f"{label}\n")
        return

    fids = [x.strip() for x in open(file_list).readlines()]
    assert len(fids) == len(labels), f"{len(fids)=} but {len(labels)=}"
    nfs_audio, nfs_video = [x.strip() for x in open(nframes_audio_file).readlines()], [x.strip() for x in open(nframes_video_file).readlines()]
    train_all, train_sub, valid, test = [], [], [], []
    for fid, (part, label), nf_audio, nf_video in zip(fids, labels, nfs_audio, nfs_video):
        # print(part)
        if part == 'test':
            test.append([fid, label, nf_audio, nf_video])
        else:
            if part == 'val':
                valid.append([fid, label, nf_audio, nf_video])
            else:
                train_all.append([fid, label, nf_audio, nf_video])
                if part == 'train':
                    train_sub.append([fid, label, nf_audio, nf_video])
    dir_30h = f"{args.lrs2}/30h_data"
    print(f"Set up 30h dir")
    os.makedirs(dir_30h, exist_ok=True)
    setup_target(dir_30h, train_sub, valid, test)
    dir_224h = f"{args.lrs2}/224h_data"
    print(f"Set up 224h dir")
    os.makedirs(dir_224h, exist_ok=True)
    setup_target(dir_224h, train_all, valid, test)
    return


if __name__ == '__main__':
    main()
