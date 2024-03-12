# AV-HuBERT Data Preprocessing

This folder contains scripts for data preparation for LRS3 and VoxCeleb2 datasets, as well as audio noise preparation (for noisy environment simulation).

## Installation
To preprocess, you need some additional packages:
```
pip install -r requirements.txt
```

## LRS3 Preprocessing

Download and decompress the [data](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html). Assume the data directory is `${lrs3}`, which contains three folders (`pretrain,trainval,test`). Follow the steps below:

### 1. Data preparation
```sh
python lrs3_prepare.py --lrs3 ${lrs3} --ffmpeg /path/to/ffmpeg --rank ${rank} --nshard ${nshard} --step ${step}
```
This will generate a list of file-ids (`${lrs3}/file.list`) and corresponding text labels (`${lrs3}/label.list`). Specifically, it includes 4 steps, where `${step}` ranges from `1,2,3,4`. Step 1, split long utterances in LRS3 `pretraining` into shorter utterances, generate their time boundaries and labels. Step 2, trim videos and audios according to the new time boundary. Step 3, extracting audio for trainval and test split. Step 4, generate a list of file ids and corresponding text transcriptions.  `${nshard}` and `${rank}` are only used in step 2 and 3. This would shard all videos into `${nshard}` and processes `${rank}`-th shard, where rank is an integer in `[0,nshard-1]`. 


### 2. Detect facial landmark and crop mouth ROIs:
```sh
python detect_landmark.py --root ${lrs3} --landmark ${lrs3}/landmark --manifest ${lrs3}/file.list \
 --cnn_detector /path/to/dlib_cnn_detector --face_detector /path/to/dlib_landmark_predictor --ffmpeg /path/to/ffmpeg \
 --rank ${rank} --nshard ${nshard}
```
```sh
python align_mouth.py --video-direc ${lrs3} --landmark ${landmark_dir} --filename-path ${lrs3}/file.list \
 --save-direc ${lrs3}/video --mean-face /path/to/mean_face --ffmpeg /path/to/ffmpeg \
 --rank ${rank} --nshard ${nshard}
```

This generates mouth ROIs in `${lrs3}/video`. It shards all videos in `${lrs3}/file.list` into `${nshard}` and generate mouth ROI for `${rank}`-th shard , where rank is an integer in `[0,nshard-1]`. The face detection and landmark prediction are done using [dlib](https://github.com/davisking/dlib). The links to download `cnn_detector`, `face_detector`, `mean_face` can be found in the help message

### 3. Count number of frames per clip
```sh
python count_frames.py --root ${lrs3} --manifest ${lrs3}/file.list --nshard ${nshard} --rank ${rank}
```
This counts number of audio/video frames for `${rank}`-th shard and saves them in `${lrs3}/nframes.audio.${rank}` and `${lrs3}/nframes.video.${rank}` respectively. Merge shards by running:

```
for rank in $(seq 0 $((nshard - 1)));do cat ${lrs3}/nframes.audio.${rank}; done > ${lrs3}/nframes.audio
for rank in $(seq 0 $((nshard - 1)));do cat ${lrs3}/nframes.video.${rank}; done > ${lrs3}/nframes.video
```

If you are on slurm, the above commands (counting per shard + merging) can be combined by:
```sh
python count_frames_slurm.py --root ${lrs3} --manifest ${lrs3}/file.list --nshard ${nshard} \
 --slurm_partition ${slurm_partition}
```
It has dependency on [submitit](https://github.com/facebookincubator/submitit) and will directly  generate `${lrs3}/nframes.audio` and `${lrs3}/nframes.video`.

### 4. Set up data directory
```sh
python lrs3_manifest.py --lrs3 ${lrs3} --manifest ${lrs3}/file.list \
 --valid-ids /path/to/valid --vocab-size ${vocab_size}
```

This sets up data directory of trainval-only (~30h training data) and pretrain+trainval (~433h training data). It will first make a tokenizer based on sentencepiece model and set up target directory containing `${train|valid|test}.{tsv|wrd}`. `*.tsv` are manifest files and `*.wrd` are text labels.  `/path/to/valid` contains held-out clip ids used as validation set.