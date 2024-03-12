# AV-HuBERT Data Preprocessing

This folder contains scripts for data preparation for LRS2 dataset following AV-HuBERT pipeline for LRS3 preprocessing.

## Installation
To preprocess, you need some additional packages:
```
pip install -r requirements.txt
```

## LRS2 Preprocessing

Download and decompress the [data](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html). Assume the data directory is `${lrs2}`, which contains two folders in `mvlrs_v1` (`pretrain,main`). Follow the steps below:

### 1. Data preparation
```sh
python lrs2_prepare.py --lrs2 ${lrs2} --ffmpeg /path/to/ffmpeg --rank ${rank} --nshard ${nshard} --step ${step}
```
This will generate a list of file-ids (`${lrs2}/mvlrs_v1/file.list`) and corresponding text labels (`${lrs2}/mvlrs_v1/label.list`). Specifically, it includes 4 steps, where `${step}` ranges from `1,2,3,4`. Step 1, split long utterances in LRS2 `pretraining` into shorter utterances, generate their time boundaries and labels. Step 2, trim videos and audios according to the new time boundary. Step 3, extracting audio for train/val/test split in directory main. Step 4, generate a list of file ids and corresponding text transcriptions according to short-pretrain in step 1 and train/val/test.txt in `${lrs2}`. Step 2 and 3 deals with pretrain and other splits respectively and can therefore be done in parallel.  `${nshard}` and `${rank}` are only used in step 2 and 3. This would shard all videos into `${nshard}` and processes `${rank}`-th shard, where rank is an integer in `[0,nshard-1]`. 


### 2. Detect facial landmark and crop mouth ROIs:
landmark_dir=${lrs2}/mvlrs_v1/landmark
```sh
python detect_landmark.py --root ${lrs2}/mvlrs_v1 --landmark ${landmark_dir} --manifest ${lrs2}/mvlrs_v1/file.list \
 --cnn_detector /path/to/dlib_cnn_detector --face_detector /path/to/dlib_landmark_predictor --ffmpeg /path/to/ffmpeg \
 --nshard ${nshard}
```
```sh
python align_mouth.py --video-direc ${lrs2}/mvlrs_v1 --landmark ${landmark_dir} --filename-path ${lrs2}/mvlrs_v1/file.list \
 --save-direc ${lrs2}/mvlrs_v1/video --mean-face /path/to/mean_face --ffmpeg /path/to/ffmpeg \
 --nshard ${nshard}
```

This generates mouth ROIs in `${lrs2}/mvlrs_v1/file.list`. It shards all videos in `${lrs2}/mvlrs_v1/file.list` into `${nshard}` and generate mouth ROI. The face detection and landmark prediction are done using [dlib](https://github.com/davisking/dlib). The links to download `cnn_detector`, `face_detector`, `mean_face` can be found in the help message

### 3. Count number of frames per clip
```sh
python count_frames.py --root ${lrs2}/mvlrs_v1 --manifest ${lrs2}/mvlrs_v1/file.list --nshard ${nshard} --rank ${rank}
```
This counts number of audio/video frames for `${rank}`-th shard and saves them in `${lrs2}/nframes.audio.${rank}` and `${lrs2}/nframes.video.${rank}` respectively. Merge shards by running:

```
for rank in $(seq 0 $((nshard - 1)));do cat ${lrs2}/mvlrs_v1/nframes.audio.${rank}; done > ${lrs2}/mvlrs_v1/nframes.audio
for rank in $(seq 0 $((nshard - 1)));do cat ${lrs2}/mvlrs_v1/nframes.video.${rank}; done > ${lrs2}/mvlrs_v1/nframes.video
```

### 4. Set up data directory
```sh
python lrs2_manifest.py --lrs2 ${lrs2}
```

This sets up data directory of train-only (~30h training data) and pretrain+train (~224h training data) which share the same val/test split. It will set up target directory containing `${train|valid|test}.{tsv|wrd}`. `*.tsv` are manifest files and `*.wrd` are text labels.