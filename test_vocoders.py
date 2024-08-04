import json
import logging
import os
import sys
import time
from tqdm import tqdm
from dataset.eer.eer_dataset import AudioPairDataset, UnitDataset
from contrastive.metrics import EERMetric
from env import AttrDict
import speaker_encoder.inference as corentinJEncoder
from pathlib import Path
import torch
from torch.utils.data.dataloader import DataLoader
from models import Generator
from constants import HIFIGAN_NO_GRAD, UNIT_HIFIGAN_NO_GRAD
import argparse
from utils import unwrap_module_generator
import torchaudio.transforms as transforms
import torch.nn.functional as F


logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
logging.getLogger(__name__)


se_path = Path("/data1/yfliu/model/CorentinJ/encoder.pt")
audio_pair_path = "/data1/yfliu/voxceleb2/voxceleb2_audiotestpairs.txt"
unit_pair_path = "/data1/yfliu/voxceleb2/voxceleb2_unittestpairs.txt"
unit_path = "/data1/yfliu/voxceleb2/all_data/test_cluster_label.km"
rank = 0
bs = 16
max_audio_sample_size = 4*16000
mel_bins = 128


class LogMelSpectrogram(torch.nn.Module):
    def __init__(self, n_fft=1024, num_mels=mel_bins, hop_size=160, win_size=1024):
        super().__init__()
        self.n_fft=n_fft
        self.hop_size=hop_size
        self.melspctrogram = transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=self.n_fft,
            win_length=win_size,
            hop_length=self.hop_size,
            center=False,
            power=1.0,
            norm="slaney",
            onesided=True,
            n_mels=num_mels,
            mel_scale="slaney",
        )

    def forward(self, wav):
        wav = F.pad(wav, ((self.n_fft-self.hop_size) // 2, (self.n_fft-self.hop_size) // 2), "reflect")
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("option", type=int, choices=[0, 1,])
    args = parser.parse_args()
    mapping_dict = {
        0:{
            "mode": HIFIGAN_NO_GRAD,
            "hcfg": "conf/hifigan/video2speech_template.json",
            # "ckpt": "/home/yfliu/16k-hifi-gan/ls_hifigan_128bin/model-400000.pt",
            "ckpt": "/home/yfliu/hifi-gan/16khifigan/mel/model-best.pt",
        },
        1:{
            "mode": UNIT_HIFIGAN_NO_GRAD,
            "hcfg": "conf/hifigan/video2speech_revise_original.json",
            "ckpt": "/home/yfliu/16k-hifi-gan/unit_hifigan/model-400000.pt",
        }
    }
    mapping_item = mapping_dict[args.option]
    args.mode = mapping_item["mode"]
    args.hifigan_config = mapping_item["hcfg"]
    with open(args.hifigan_config) as f:
        data = f.read()
    json_config = json.loads(data)
    default_nones = {
        'prosody_type', 'st_type', 'hu_repr_name',
        'unit_name', 'valid_unit_name', 'test_unit_name',
    }
    json_config.update({k:None for k in default_nones if k not in json_config.keys()})
    h = AttrDict(json_config)
    args.hifigan_ckpt = mapping_item["ckpt"]
    return args, h


if __name__ == '__main__':
    torch.cuda.set_device(rank)  # A very strong boost. See https://github.com/jik876/hifi-gan/pull/25
    device = torch.device('cuda:{:d}'.format(rank))
    corentinJEncoder.load_model(se_path, device)
    args, h = get_args()
    if args.mode == HIFIGAN_NO_GRAD:
        mel_dim = h.num_mels
        vocoder = Generator(h, mel_dim)
    elif args.mode == UNIT_HIFIGAN_NO_GRAD:
        n_units = h.k
        n_units += 1
        vocoder = Generator(
            h, 
            h.num_mels, 
            unit_nums=h.k
            )
    vocoder.to(device).eval()
    if args.hifigan_ckpt is not None:
        # hifigan ckpt is not supposed to be updated in training with 2mel mode
        hifigan_weight = torch.load(args.hifigan_ckpt, map_location=device)
        vocoder.load_state_dict(unwrap_module_generator(
            hifigan_weight["generator"]["model"], ignore_conv_pre=False,
            ))
    eer_metric = EERMetric(rank)
    logmel = LogMelSpectrogram().to(device)
    if args.mode == HIFIGAN_NO_GRAD:
        audiopair_dataset = AudioPairDataset(pair_path=audio_pair_path)
    elif args.mode == UNIT_HIFIGAN_NO_GRAD:
        audiopair_dataset = UnitDataset(unit_path=unit_path, pair_path=unit_pair_path)
    dataloader = DataLoader(audiopair_dataset, batch_size=bs, collate_fn=audiopair_dataset.collater)
    pbar = tqdm(dataloader)
    logging.info("start")
    for batch in pbar:
        time_start = time.perf_counter()
        is_positive, audios, padding_mask = batch
        padding_mask = padding_mask[:, :max_audio_sample_size]
        min_length = padding_mask.sum(dim=-1).min().item()
        audios = audios[:, :min_length]
        if args.mode == HIFIGAN_NO_GRAD:
            y_mel = logmel(audios.to(device))
            y_mel = y_mel.transpose(-1, -2)  # [B, T, mel]
            with torch.no_grad():
                audios_generated = vocoder(y_mel)
        elif args.mode == UNIT_HIFIGAN_NO_GRAD:
            units = audios.to(device)
            with torch.no_grad():
                audios_generated = vocoder(units)
        similarity = corentinJEncoder.compute_similarity(audios_generated.squeeze(1), padding_mask, max_audio_sample_size, pad_audio=False)
        eer_metric.update(
                preds=similarity,
                labels=is_positive,
            )
        pbar.set_description(f'time past={time.perf_counter() - time_start}')
        logging.info(f"{is_positive.tolist()}, {similarity.tolist()}")
        try:
            curr_eer = eer_metric.compute()
            pbar.set_description(f'{curr_eer=}')
        except ValueError as e:
            pbar.set_description("got nan, continuing")
    logging.warn(eer_metric.compute())
