# DiViSe: Direct Visual-Input Speech Synthesis

We proposed DiViSe, a Video-to-Speech Synthesis framework that resynthesizes audio waveforms from silent videos. The code and weights are open-sourced in this repository for reproduction purposes. Samples generated with HiFi-GAN and Griffin-Lim are provided in supplementary materials too.

**Abstract :**
The field of Video-to-Speech (V2S) synthesis aims to convert silent lip movements into audible speech using audio-visual data. Traditional V2S methods required the incorporation of speaker information, risking potential knowledge leakage. This study introduces a novel V2S synthesis technique utilizing the pretrained audio-visual model for direct Mel-spectrogram prediction without requiring any input beyond silent video during both training and inference. When coupled with an off-the-shelf ASR system, the audio produced by our approach achieves the lowest Word Error Rate (WER) compared to other existing V2S synthesis methods that predict Mel-spectrograms, thereby demonstrating the high intelligibility of the generated audio. Additionally, this method sets a new state-of-the-art in STOI and ESTOI metrics on the LRS2 and LRS3 datasets. We tested conditions where constraints are imposed, such as operating in low-resource settings or employing models with reduced sizes, where our method still proved to show strong performance relative to existing approaches. Our approach offers a promising direction toward accurate speech reconstruction from silent videos. Code and model parameters will be made publicly accessible upon the acceptance of this paper.

## What we will open source

1. DiViSe implementation and data preprocessing scripts (This repo)
1. ReVISE implementation (This repo)
1. Scripts to train vocoders on resampled LJSpeech (16kHz). This is given in our 16k-hifigan repo and includes:
    - resampling script (Thanks to [bshall's implementation](https://github.com/bshall/hifigan))
    - training scripts for:
        - HiFi-GAN
        - Unit-HiFiGAN (Required for ReVISE to generate audio)
1. Generated video examples (examples.zip).
1. (In the future) Pretrained weights listed in Section **Pretrained Model**.

## Pre-requisites
1. Python 3.8
1. Several NVIDIA GPUs (RTX 3090 or 4090 will be fine in my case). 
    -  4 are required to run low-resource setting.
    -  8 are required to run full-resource setting.
    - Please use the number of GPUs strictly as the number of updates will differ from my training setting if a different setting of GPUs is set. This is because currently I have only set the number of updates **per GPU** in my current setting.
1. Install python requirements. Please refer to [requirements.txt](requirements.txt).
1. Have a pretrained vocoder: 
    - DiViSe: This is optional. Griffin-Lim is always enabled even if no vocoder is used.
    - ReVISE: You should have a pretrained Unit-HiFiGAN model.

## Data Preparation
1. (LRS3) Please refer to [AV-HuBERT](https://github.com/facebookresearch/av_hubert/tree/258fb50e155134eec2c4b49c2ae8de267075fd18/avhubert/preparation).
1. (LRS2) See [dataset/lrs2/README.md](dataset/lrs2/README.md) for instructions. Actually this is identical to AV-HuBERT's preprocessing except the modifications we made to suit LRS2's file structure.

Guidance for LJSpeech preprocessing is given in our 16k-hifigan repo.

## Pretrained Model
We plan to provide pretrained weights on Huggingface after anonymous period.

The weights we plan to provide are as follows for your reference. If you would like to see any other pretrained weights, just let me know and I'll let them on shelf.

### V2S Models
|Model|Dataset|Updates (per GPU)|# of GPUs used|
|------|---|---|---|
|DiViSe|LRS3|45000|8|
|DiViSe-BASE|LRS3|11250|4|
|DiViSe|LRS2|45000|8|
|DiViSe-BASE|LRS2|11250|4|
|ReVISE (Our Implementation)|LRS2|45000|8|

### Vocoders
All vocoders are trained on resampled version (16kHz) of LJSpeech Dataset. See [vocoders/README.md](vocoders/README.md)
|Models|
|:------:|
|HiFiGAN|
|BigVGAN-base|
|Parallel WaveGAN (PWG)|
|Unit-HiFiGAN (For ReVISE implementation)|

Griffin-Lim is already implemented in this repository.

## Training

### Configuration Setup
- conf/avhubert:
    - You will need to modify the config file you need to run. Update `task:data` to your preprocessed data directory.
    - Either preprocessed dir of LRS3 and LRS2 will be fine to fit `task:data`.
        - For low-resource setting, fit `30h` dir of your preprocessed dataset.
        - For full-resource setting, fit `433h` dir / `224h` dir of your preprocessed dataset.

- conf/hifigan:
    - For DiViSe, do noting.
    - For ReVISE implementation, modify `unit_name`, `valid_unit_name` and `test_unit_name` and `k` to your specification. One should refer to [hubert/README.md](hubert/README.md) first before training ReVISE.
    - Normally you do not need to modify `total_updates`. Just make sure you're using the right number of GPUs.

Script references are given as listed below. Note that number of GPUs needed must match to give reproducable results.

### Evironment Variables
We are taking HiFi-GAN as our default vocoder here. One may also try BigVGAN and PWG with `--bigvgan_ckpt` and `--pwg_ckpt`.
```
your_ckpt=[REPLACE HERE]
your_avhb_cfg=[REPLACE HERE]
your_avhubert_ckpt=[REPLACE HERE]  # base_lrs3_iter5.pt for BASE setting and large_vox_iter5.pt for LARGE setting (default).
your_hifigan_ckpt=[REPLACE HERE]  # One can use more vocoders other than HiFi-GAN for DiViSe. See train.py's argparser for more information. For ReVISE, ensure you're using Unit-HiFiGAN.
```
### Training Commands

`--hifigan_ckpt` is optional for DiViSe. You may also swap to `--bigvgan_ckpt` and `--pwg_ckpt` for DiViSe.
|Command|Model|
|----|---|
`python train.py --checkpoint_path output/baseline/$your_ckpt --hifigan_config conf/hifigan/video2speech_template.json --avhubert_config conf/avhubert/${your_avhb_cfg}.yaml --avhubert_ckpt $your_avhubert_ckpt --hifigan_ckpt $your_hifigan_ckpt --wandb`|DiViSe|
`python train.py --checkpoint_path output/revise/$your_ckpt --hifigan_config conf/hifigan/video2speech_revise_original.json --avhubert_config conf/avhubert/${your_avhb_cfg}.yaml --avhubert_ckpt $your_avhubert_ckpt --hifigan_ckpt $your_hifigan_ckpt --wandb`|ReVISE|


## Acknowledgements
Special thanks to [HiFi-GAN](https://github.com/jik876/hifi-gan) and [AV-HuBERT](https://github.com/facebookresearch/av_hubert/), where this repository is built upon. We also appreciate all other works mentioned in this repository.

