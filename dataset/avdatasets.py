# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import pdb
import sys
import time
from typing import Any, List, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
from python_speech_features import logfbank
from scipy.io import wavfile
from dataset.meldataset import MAX_WAV_VALUE, load_wav
from librosa.util import normalize as librsa_normalize

DBG=True

import dataset.transform_custom_utils as custom_utils

logger = logging.getLogger(__name__)


def load_audio_visual_simple(manifest_path, max_keep, min_keep):
    n_long, n_short = 0, 0
    names, inds, sizes = [], [], []

    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            sz = int(items[-2]) # 
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                video_path = items[1]
                audio_path = items[2]
                audio_id = items[0]
                names.append((video_path, audio_path+':'+audio_id))
                inds.append(ind)
                sizes.append(sz)
    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, names, inds, tot, sizes

def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets

def load_km_labels(km_path, inds, tot):
    with open(km_path, 'r') as f:
        km_labels = f.readlines()
        assert len(km_labels) == tot, f"{len(km_labels)=} does not match lines in tsv files({tot})." \
                    "Please check if they are on the same split."
        km_labels = [km_labels[i] for i in inds]
    return km_labels

class AVHubertDataset(FairseqDataset):
    def __init__(
            self,
            manifest_path: str,
            sample_rate: float,
            label_paths: List[str],
            max_keep_sample_size: Optional[int] = None,
            min_keep_sample_size: Optional[int] = None,
            max_sample_seconds: Optional[float] = None,
            pitch_type: Optional[str] = None,  # Should be pyworld or kaldi
            km_path: Optional[str] = None,  # Should be path to your .km file
            km_pad_class: Optional[int] = None,  # This will be neccessary in end-to-end training
            hu_name: Optional[str] = None,  # Should be {kmeans_split} in hubertrep_export.py 
            fake_km_mask: bool = False,  # provide km padding mask even on a set without km labels.
            shuffle: bool = True,
            pad_audio: bool = False,
            normalize: bool = False,
            store_labels: bool = True,
            random_crop: bool = False,
            single_target: bool = False,
            image_mean: float=0,
            image_std: float=1,
            image_crop_size: int=88,
            image_aug: bool=False,
            modalities: Optional[List[str]]=None,
            noise_fn=None,
            noise_prob=0,
            noise_snr=0,
            noise_num=1
    ):
        self.modalities = set(modalities)  # should always be {'video', 'audio'}
        self.audio_root, self.names, inds, tot, self.sizes = load_audio_visual_simple(manifest_path, max_keep_sample_size, min_keep_sample_size)
        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.random_crop = random_crop

        self.num_labels = len(label_paths)
        self.single_target = single_target
        self.store_labels = store_labels
        self.noise_wav, self.noise_prob, self.noise_snr, self.noise_num = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else [], noise_prob, noise_snr, noise_num
        if store_labels:
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]

        self.max_sample_seconds = (
            max_sample_seconds if max_sample_seconds is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        self.pitch_type = pitch_type
        self.hu_name = hu_name
        self.fake_km_mask = fake_km_mask
        self.km_pad_idx = km_pad_class
        if km_path:
            # km_label is stored in a single text-format file so it must be preloaded into running memory.
            self.km_labels = load_km_labels(km_path, inds, tot)
        else:
            self.km_labels = None
        if image_aug:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize( 0.0,255.0 ),
                custom_utils.RandomCrop((image_crop_size, image_crop_size)),
                custom_utils.HorizontalFlip(0.5),
                custom_utils.Normalize(image_mean, image_std) ])
        else:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize( 0.0,255.0 ),
                custom_utils.CenterCrop((image_crop_size, image_crop_size)),
                custom_utils.Normalize(image_mean, image_std) ])
        self.sr_video = 25
        self.sr_audio = 16000
        self.hop_size_mel = 160  # for trimming purpose only
        self.sr_pitch = self.sr_audio//self.hop_size_mel
        assert self.sr_audio%self.hop_size_mel==0 and self.sr_pitch%self.sr_video==0, "Please check sample rate"
        self.video2mel_magnitude = int(self.sr_audio/self.hop_size_mel/self.sr_video)  # 4
        self.max_video_sample_size = int(self.sr_video*self.max_sample_seconds)
        self.max_audio_sample_size = int(self.sr_audio*self.max_sample_seconds)
        self.sr_pitch = self.video2mel_magnitude * self.sr_video
        self.max_pitch_sample_size = int(self.sr_pitch*self.max_sample_seconds)
        self.sr_km = 50  # HuBERT output is 50 Hz
        self.max_km_sample_size = int(self.sr_km*self.max_sample_seconds)
        logger.info(f"using video fps {self.sr_video} and audio sr {self.sr_audio}.")
        logger.info(f"image transform: {self.transform}")

        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, {self.max_sample_seconds=}, ")
        logger.info(
            f"Noise wav: {noise_fn}->{len(self.noise_wav)} wav, Prob: {self.noise_prob}, SNR: {self.noise_snr}, Number of mixture: {self.noise_num}"
        )

    def get_label(self, index, label_idx):
        if self.store_labels:
            label = self.label_list[label_idx][index]
        else:
            with open(self.label_paths[label_idx]) as f:
                offset_s, offset_e = self.label_offsets_list[label_idx][index]
                f.seek(offset_s)
                label = f.read(offset_e - offset_s)

        # if self.label_processors is not None:
        #     label = self.label_processors[label_idx](label)
        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]
    
    def load_everything(self, index):
        """
        Load video feature and raw audio waveform
        Returns:
        video_feats: numpy.ndarray of shape [T, H, W, 1], audio_feats: numpy.ndarray of shape [T]
        """
        mix_name = self.names[index]
        video_fn, audio_fn = mix_name
        audio_fn, audio_id = audio_fn.split(':')
        audio_id = audio_id.split('/')[-1]
        audio_base_dir = os.path.dirname(audio_fn)
        if self.pitch_type == 'pyworld':
            pw_pitch_fn = os.path.join(audio_base_dir, f"{audio_id}_pw_dio.npy")
            pitch = np.load(pw_pitch_fn)
        elif self.pitch_type == 'kaldi':
            kaldi_pitch_fn = os.path.join(audio_base_dir, f"{audio_id}_kaldi.pt")
            pitch = torch.load(kaldi_pitch_fn)
        else:
            pitch = None
        if self.km_labels is not None:
            km_labels = [int(x) for x in self.km_labels[index].strip().split(' ')]  # 50 Hz
        else:
            km_labels = None
        if self.hu_name is not None:
            load_path = os.path.join(audio_base_dir, f"{audio_id}_{self.hu_name}.npy")
            hubert_hu = np.load(load_path)
        else:
            hubert_hu = None
        if 'video' in self.modalities:
            video_feats = self.load_video(video_fn) # [T, H, W, 1]
        else:
            video_feats = None
        if 'audio' in self.modalities:
            wav_data, sample_rate = load_wav(audio_fn)
            assert sample_rate == 16_000 and len(wav_data.shape) == 1
            wav_data = wav_data / MAX_WAV_VALUE
            wav_data = librsa_normalize(wav_data) * 0.95
            if np.random.rand() < self.noise_prob:
                wav_data = self.add_noise(wav_data)  # noise_prob is 0, don't worry.
        else:
            wav_data = None
        return video_feats, wav_data, pitch, km_labels, hubert_hu

    def load_video(self, audio_name):
        feats = custom_utils.load_video(os.path.join(self.audio_root, audio_name))
        feats = self.transform(feats)
        feats = np.expand_dims(feats, axis=-1)
        return feats

    def select_noise(self):
        rand_indexes = np.random.randint(0, len(self.noise_wav), size=self.noise_num)
        noise_wav = []
        for x in rand_indexes:
            noise_wav.append(wavfile.read(self.noise_wav[x])[1].astype(np.float32))
        if self.noise_num == 1:
            return noise_wav[0]
        else:
            min_len = min([len(x) for x in noise_wav])
            noise_wav = [x[:min_len] for x in noise_wav]
            noise_wav = np.floor(np.stack(noise_wav).mean(axis=0))
            return noise_wav

    def add_noise(self, clean_wav):
        clean_wav = clean_wav.astype(np.float32)
        noise_wav = self.select_noise()
        if type(self.noise_snr) == int or type(self.noise_snr) == float:
            snr = self.noise_snr
        elif type(self.noise_snr) == tuple:
            snr = np.random.randint(self.noise_snr[0], self.noise_snr[1]+1)
        clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
        if len(clean_wav) > len(noise_wav):
            ratio = int(np.ceil(len(clean_wav)/len(noise_wav)))
            noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
        if len(clean_wav) < len(noise_wav):
            start = 0
            noise_wav = noise_wav[start: start + len(clean_wav)]
        noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1))
        adjusted_noise_rms = clean_rms / (10**(snr/20))
        adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
        mixed = clean_wav + adjusted_noise_wav

        #Avoid clipping noise
        max_int16 = np.iinfo(np.int16).max
        min_int16 = np.iinfo(np.int16).min
        if mixed.max(axis=0) > max_int16 or mixed.min(axis=0) < min_int16:
            if mixed.max(axis=0) >= abs(mixed.min(axis=0)): 
                reduction_rate = max_int16 / mixed.max(axis=0)
            else :
                reduction_rate = min_int16 / mixed.min(axis=0)
            mixed = mixed * (reduction_rate)
        mixed = mixed.astype(np.int16)
        return mixed

    def __getitem__(self, index):
        video_feats, wav_data, pitch_data, km_labels, hubert_hu = self.load_everything(index)
        wav_data, video_feats = torch.FloatTensor(wav_data) if wav_data is not None else None, torch.from_numpy(video_feats.astype(np.float32)) if video_feats is not None else None
        if pitch_data is not None:
            pitch_data = torch.FloatTensor(pitch_data)
        if km_labels is not None:
            km_labels = torch.LongTensor(km_labels)
        if hubert_hu is not None:
            hubert_hu = torch.Tensor(hubert_hu)
        labels = self.get_labels(index)
        fid = self.names[index][1].split(':')[1]
        return {"id": index, 'fid': fid, "video_source": video_feats, 'audio_source': wav_data, "label_list": labels,
                "pitch_source": pitch_data, "km_source": km_labels, "hubert_source": hubert_hu,}

    def __len__(self):
        return len(self.sizes)

    def crop_to_max_size(self, wav, target_size, start=None):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0
        # longer utterances
        if start is None:
            start, end = 0, target_size
            if self.random_crop:
                start = np.random.randint(0, diff + 1)
                end = size - diff + start
        else:
            end = start + target_size
        return wav[start:end], start

    def collater(self, samples):
        samples = [s for s in samples if s["id"] is not None]
        if len(samples) == 0:
            return {}
        audio_source, video_source = [s["audio_source"] for s in samples], [s["video_source"] for s in samples]
        pitch_source = [s["pitch_source"] for s in samples]
        km_source = [s["km_source"] for s in samples]
        hu_source = [s["hubert_source"] for s in samples]
        with_pitch = None not in pitch_source
        with_km = None not in km_source
        with_hu = None not in hu_source
        if audio_source[0] is None:
            audio_source = None
        if video_source[0] is None:
            video_source = None
        if audio_source is not None:
            audio_sizes = [len(s) for s in audio_source]
            if with_pitch:
                pitch_sizes = [len(s) for s in pitch_source]
            if with_km:
                km_sizes = [len(s) for s in km_source]
            if with_hu:
                km_sizes = [len(s) for s in hu_source]
        if video_source is not None:
            video_sizes = [len(s) for s in video_source]
        if audio_source is not None and video_source is not None:
            # compulsory align and trim for audio
            audio_sizes = [video_size*self.video2mel_magnitude*self.hop_size_mel for video_size in video_sizes]
            if with_pitch:
                pitch_sizes = [video_size*self.video2mel_magnitude for video_size in video_sizes]
            if with_km or with_hu:
                km_sizes = [video_size*2 for video_size in video_sizes]  #  only works when video is 25 Hz -> 50 Hz in HuBERT
        if self.pad_audio:
            func = lambda curr_x, max_sample_x: min(max(curr_x), max_sample_x)
        else:
            func = lambda curr_x, max_sample_x: min(min(curr_x), max_sample_x)
        collated_pitches = None
        collated_km = None
        collated_hu = None
        padding_mask_km = None
        if audio_source is not None:
            audio_size = func(audio_sizes, self.max_audio_sample_size)
            collated_audios, padding_mask, audio_starts = self.collater_wav(audio_source, audio_size)
            second_starts = [audio_start/self.sr_audio for audio_start in audio_starts]  # By default(padding), This will always be 0.0.
            if with_pitch:
                pitch_size = func(pitch_sizes, self.max_pitch_sample_size)
                pitch_starts = [int(second_start*self.sr_pitch) for second_start in second_starts]
                collated_pitches, _, pitch_starts = self.collater_wav(pitch_source, pitch_size, pitch_starts)
            if with_km or with_hu:
                km_size = func(km_sizes, self.max_km_sample_size)
                km_starts = [int(second_start*self.sr_km) for second_start in second_starts]
                if with_km:
                    collated_km, padding_mask_km, km_starts = self.collater_wav(km_source, km_size, km_starts, 
                                                                                pad_value=self.km_pad_idx if self.km_pad_idx is not None else 0.0)
                if with_hu:
                    collated_hu, padding_mask_km, km_starts = self.collater_wav(hu_source, km_size, km_starts)
            elif self.fake_km_mask:
                # for sets without labels, make padding mask available
                km_sizes = [video_size*2 for video_size in video_sizes]
                fake_km_source = [torch.zeros(t) for t in km_sizes]
                km_size = func(km_sizes, self.max_km_sample_size)
                km_starts = [int(second_start*self.sr_km) for second_start in second_starts]
                _, padding_mask_km, km_starts = self.collater_wav(fake_km_source, km_size, km_starts)
        else:
            collated_audios, audio_starts = None, None
        if video_source is not None:
            video_size = func(video_sizes, self.max_video_sample_size)
            video_starts = [int(second_start*self.sr_video) for second_start in second_starts]
            collated_videos, padding_mask_mel, video_starts = self.collater_video_with_mel_mask(video_source, video_size, video_starts)
        else:
            collated_videos = None
        targets_by_label = [
            [s["label_list"][i] for s in samples]
            for i in range(self.num_labels)
        ]
        targets_list, lengths_list, ntokens_list = self.collater_label_text(targets_by_label)
        source = {"audio": collated_audios, "video": collated_videos, "pitch": collated_pitches, "km": collated_km, "hu": collated_hu,}
        net_input = {"source": source,  # Definitely not None
                    "padding_mask_wav": padding_mask,  # Definitely not None
                    "padding_mask_mel": padding_mask_mel,  # Definitely not None
                    "padding_mask_km": padding_mask_km,  # Could be None (e.g. in validation)
                    }  # padding_mask_wav is for waveform(16000Hz), _mel for mel spectrogram(100Hz), _km 50Hz
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
            "utt_id": [s['fid'] for s in samples]
        }

        if self.single_target:
            batch["target_lengths"] = lengths_list[0]
            batch["ntokens"] = ntokens_list[0]
            batch["target"] = targets_list[0]
        else:
            batch["target_lengths_list"] = lengths_list
            batch["ntokens_list"] = ntokens_list
            batch["target_list"] = targets_list
        return batch

    def collater_video_with_mel_mask(self, videos, video_size, video_starts=None):
        video_feat_shape = list(videos[0].shape[1:])
        collated_videos = videos[0].new_zeros([len(videos), video_size]+video_feat_shape)
        padding_mask_mel = (
            torch.BoolTensor(len(videos), video_size*4).fill_(False) # A HARD-CODED 4x ratio!!!
        )
        start_known = video_starts is not None
        video_starts = [0 for _ in videos] if not start_known else video_starts
        for i, video in enumerate(videos):
            diff = len(video) - video_size
            if diff == 0:
                collated_videos[i] = video
            elif diff < 0:
                assert self.pad_audio
                collated_videos[i] = torch.cat(
                    [video, video.new_full([-diff]+video_feat_shape, 0.0)]
                )
                padding_mask_mel[i, diff*4:] = True
            else:
                collated_videos[i], video_starts[i] = self.crop_to_max_size(
                    video, video_size, video_starts[i] if start_known else None
                )
        collated_videos = collated_videos.permute((0, 4, 1, 2, 3)).contiguous() # [B, T, H, W, C] -> [B, C, T, H, W]
        return collated_videos, padding_mask_mel, video_starts
    
    def collater_wav(self, wavs, wav_size, wav_starts=None, pad_value=0.0):
        wav_feat_size = list(wavs[0].shape[1:])  # In case we have pitch as input. Raw waveforms doesn't need this.
        collated_size = [len(wavs), wav_size]+wav_feat_size
        collated_wavs = wavs[0].new_zeros(collated_size)  # [B, T] for wav or [B, T, 2] for kaldi pitch
        padding_mask = (
            torch.BoolTensor(len(wavs), wav_size).fill_(False) # 
        )
        start_known = wav_starts is not None
        wav_starts = [0 for _ in wavs] if not start_known else wav_starts
        for i, wav in enumerate(wavs):
            diff = len(wav) - wav_size
            if diff == 0:
                collated_wavs[i] = wav
            elif diff < 0:
                assert self.pad_audio
                collated_wavs[i] = torch.cat(
                    [wav, wav.new_full([-diff]+wav_feat_size, pad_value)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_wavs[i], wav_starts[i] = self.crop_to_max_size(
                    wav, wav_size, wav_starts[i] if start_known else None
                )
        return collated_wavs, padding_mask, wav_starts


    def collater_label_text(self, targets_by_label):
        targets_list, lengths_list, ntokens_list = [], [], []
        for targets in targets_by_label:
            lengths = [len(t) for t in targets]
            ntokens = sum(lengths)
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]
