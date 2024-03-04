# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import torchaudio
import torch
import tqdm
from speechtokenizer import SpeechTokenizer

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature")


class SpeechTokensReader(object):
    def __init__(self, config_path, ckpt_path, device):
        self.model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path).to(device)
        self.model.eval()
        self.device = device
        logger.info(f"model loaded from {ckpt_path} on {device=}")

    def read_audio(self, path, ref_len=None, min_seconds=-1):
        wav, sr = torchaudio.load(path)
        # monophonic checking
        if wav.shape[0] > 1:
            wav = wav[:1,:]
        actual_len = wav.shape[-1]

        if sr != self.model.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.model.sample_rate)
        if ref_len is not None and abs(ref_len - actual_len) > 160:
            logging.warning(f"ref {ref_len} != read {actual_len} ({path})")
        if min_seconds > 0:
            pad_wav_len = int(min_seconds*self.model.sample_rate-actual_len)
            if pad_wav_len > 0:
                wav = torch.cat((wav, torch.zeros(1, pad_wav_len)), dim=-1)
        wav = wav.unsqueeze(0)
        return wav.to(self.device)

    def get_feats(self, path, ref_len=None, min_seconds=-1):
        wav = self.read_audio(path, ref_len, min_seconds)
        with torch.no_grad():
            # Extract discrete codes from SpeechTokenizer
            codes = self.model.encode(wav) # codes: (n_q, B, T)
            codes = codes.squeeze()  # (n_q, T)
            return codes


def get_path_iterator(tsv):
    with open(tsv, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        tot = len(lines)
        logger.info(
            f"{tot=}"
        )

        def iterate():
            for line in lines:
                subpath, nsample = line.split("\t")
                yield f"{root}/{subpath}", int(nsample)

        return iterate, len(lines)


def dump_feature(
    tsv_dir, split, token_name, lab_dir
):
    tsv_dir = f"{tsv_dir}/{split}.tsv"
    generator, tot = get_path_iterator(tsv_dir)
    iterator = generator()
    lab_path = f"{lab_dir}/{split}.st"

    with open(lab_path, "r") as code_st:
        for code_line, (path, nsample) in zip(code_st, tqdm.tqdm(iterator, total=tot)):
            export_file_name = path.split('.')[0]+f'_{token_name}.st'
            with open(export_file_name, 'w') as f:
                f.write(code_line)
    logger.info("finished successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_dir")
    parser.add_argument("split")
    parser.add_argument("token_name")
    parser.add_argument("lab_dir")
    args = parser.parse_args()
    logger.info(args)

    dump_feature(**vars(args))
