# DiVISe: Direct Visual-Input Speech Synthesis Preserving Speaker Characteristics And Intelligibility

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2503.05223v1)
[![githubio](https://img.shields.io/static/v1?message=Audio%20Samples&logo=Github&labelColor=grey&color=blue&logoColor=white&label=%20&style=flat)](https://pussycat0700.github.io/DiVISe-Demo/)

This is the official implementation for [**DiVISe: Direct Visual-Input Speech Synthesis Preserving Speaker Characteristics And Intelligibility**](https://arxiv.org/abs/2503.05223v1).

**Abstract :**
Video-to-speech (V2S) synthesis, the task of generating speech directly from silent video input, is inherently more challenging than other speech synthesis tasks due to the need to accurately reconstruct both speech content and speaker characteristics from visual cues alone. Recently, audio-visual pre-training has eliminated the need for additional acoustic hints in V2S, which previous methods often relied on to ensure training convergence. However, even with pre-training, existing methods continue to face challenges in achieving a balance between acoustic intelligibility and the preservation of speaker-specific characteristics.
We analyzed this limitation and were motivated to introduce DiVISe (**Di**rect **V**isual-**I**nput **S**peech Synth**e**sis), an end-to-end V2S model that predicts Mel-spectrograms directly from video frames alone. Despite not taking any acoustic hints,
DiVISe effectively preserves speaker characteristics in the generated audio, and achieves superior performance on both objective and subjective metrics across the LRS2 and LRS3 datasets. Our results demonstrate that DiVISe not only outperforms existing V2S models in acoustic intelligibility but also scales more effectively with increased data and model parameters.

## What we open source in this repository:

1. DiViSe implementation and an unofficial ReVISE re-implementation for Video-to-Speech task.
1. Scripts to train vocoders on resampled LJSpeech (16kHz). This is given in [custom_hifigan](./custom_hifigan/) as a git submodule forked from [bshall's implementation](https://github.com/bshall/hifigan).
1. Pretrained parameters for DiVISe and ReVISE in our implementation, as well as their vocoders.

## Pre-requisites
1. Python 3.8
1. Install python requirements. Please refer to [requirements.txt](requirements.txt). [TODO: Current requirements.txt may be inaccurate.]
1. initialize submodule with `git submodule update --init --recursive`.
1. Replace `SPEAKER_ENCODER_PATH` with your path in [env.py](./env.py).

## Data Preparation
1. (LRS3) Please refer to [AV-HuBERT](https://github.com/facebookresearch/av_hubert/tree/258fb50e155134eec2c4b49c2ae8de267075fd18/avhubert/preparation).
1. (LRS2) See [dataset/lrs2/README.md](dataset/lrs2/README.md) for instructions. This is effectively identical to AV-HuBERT's preprocessing except the modifications we made to suit LRS2's file structure.

If you run into any problem with the data preparation pipeline, please feel free to open an issue to let us know.

## Configuration Setup

Two config files will be needed to run the code. Genrally, `conf/avhubert` handles the data location and the model size, while `conf/hifigan` contains training-related hyper parameters.

- `conf/avhubert`: Modification needed as follows:
    - Replace `???` of `task:data` with your preprocessed data directory.
    - Either preprocessed dir of LRS3 and LRS2 will be fine to fit `task:data`.
        - For low-resource setting, fit `30h` dir of your preprocessed dataset.
        - For full-resource setting, fit `433h` dir / `224h` dir of your preprocessed dataset.

- `conf/hifigan`: Modification not needed
    |Model|Config File|
    |----|---|
    |DiVISe|[conf/hifigan/video2speech_template.json](conf/hifigan/video2speech_template.json)|
    |ReVISE|[conf/hifigan/video2speech_revise_original.json](conf/hifigan/video2speech_revise_original.json)|

## Training

For strict reimplementation, please follow the number of GPUs given below. as the number of updates is set **per GPU** in our setting, the number of updates will differ with a different number of GPUs. RTX 3090 or RTX4090 will be fine in our case.
    -  4 are required to run 30h setting.
    -  8 are required to run 433h setting.

Script references are given as listed below. Here we assume we train with 433h full-resource setup with 8 GPUs.

### Evironment Variables

- your_ckpt_path=[REPLACE HERE]  # ckpt path
- your_avhubert_ckpt=[REPLACE HERE]  # base_lrs3_iter5.pt for BASE setting and large_vox_iter5.pt for LARGE - setting (default) You may find these checkpoints [here](https://facebookresearch.github.io/av_hubert/).
- your_hifigan_ckpt=[REPLACE HERE]  # One can use more vocoders other than HiFi-GAN for DiViSe. See train.py's argparser for more information. For ReVISE, ensure you're using Unit-HiFiGAN. You may find these checkpoints [here](#pretrained-model).

### Training Commands


|Command|Model|
|----|---|
`python train.py --checkpoint_path $your_ckpt_path --hifigan_config conf/hifigan/video2speech_template.json --avhubert_config conf/avhubert/large_avhubert_template.yaml --avhubert_ckpt $your_avhubert_ckpt --hifigan_ckpt $your_hifigan_ckpt --wandb`|DiViSe|
`python train.py --checkpoint_path $your_ckpt_path --hifigan_config conf/hifigan/video2speech_revise_original.json --avhubert_config conf/avhubert/large_avhubert_template.yaml --avhubert_ckpt $your_avhubert_ckpt --hifigan_ckpt $your_hifigan_ckpt --wandb`|ReVISE|


## Evaluation

Evaluation can be done by simply adding an extra `--test` argument in [Training Commands](#training-commands), with only a single GPU.

## Pretrained Model

We release the links to model parameters trained under full resource setting of LRS3 in this paper as follows.

### V2S Models
|Model|Link|
|:------:|---|
|DiViSe||
|ReVISE (Our Implementation)||

### Vocoders
All vocoders are trained on resampled version (16kHz) of LJSpeech Dataset in [custom_hifigan](./custom_hifigan/).
|Models|Link|
|:------:|---|
|HiFiGAN (Fine-tuned For DiVISe)||
|HiFiGAN (Pre-trained only)||
|Unit-HiFiGAN (For ReVISE)||


## Acknowledgements
Special thanks to [HiFi-GAN](https://github.com/jik876/hifi-gan) and [AV-HuBERT](https://github.com/facebookresearch/av_hubert/), where this repository is built upon. We also appreciate all other works mentioned in this repository.

