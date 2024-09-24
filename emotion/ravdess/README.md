# The RAVDESS dataset

Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391.

# Overview

We apply emotional speech classification task for the purpose of testing unit embeddings from different vocoders. Embeddings are fixed throughout the training and act as the only input to your classification model.

You will need the following steps to prepare yourself up for emotional speech classification task.

1. For unit-based vocoders: export KMeans label to ravdess dataset.
1. For FaRL+unit-based vocoders: export one frame of the video to picture with ffmpeg.

# Preparation

## Set Definition

We split the sets as follows:

- 01 to 20: train sets
- 21 to 22: valid sets
- 23 to 24: test sets

Here is what you need to do in detail.

## Prerequisites

Modify paths to your need in [ravdess_paths.py](ravdess_paths.py)

## KMeans label extraction

All data preprocessing scripts can be found in [fairseq](https://github.com/facebookresearch/fairseq/tree/da8fb630880d529ab47e53381c30ddc8ad235216/examples/hubert/simple_kmeans).

Detailed Hubert Kmeans clustering script is not included in this repo. One should refer to the code link above for guidance on preprocessing steps.

However, for HuBERT clustering, a tsv file is required to generate labels for each set. We provide the following scripts that help you do it.

- [resample.py](resample.py): resamples audio to 16kHz and export them to AUDIO_RESAMPLED. AUDIO_RESAMPLED_HUBERT_TSV will be also be saved.

The other scripts can be found in HuBERT repo.

## Video Frame extraction (FaRL+Unit vocoder)

See [extract_frames.py](../../dataset/eer/preprocessing/extract_frame.py).

## tsv export

See [export_tsv.py](export_tsv.py).

tsv file for each set contains lines with columns ordered as follows each separated with `\t`:

1. Frame picture path
1. Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
1. Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
1. Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
1. Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
1. Gender (01 = male, 02 = female).
1. Original wav path
1. Wav Length

kmeans labels should be saved in AUDIO_RESAMPLED/hubert.km
- See {split}_TSV for RAVDESS labels.
- See {split}_HUBERT_TSV for hubert labels. They should naturally correspond to RAVDESS labels on every line.


# Training

See [train_emotion.py](../../train_emotion.py)