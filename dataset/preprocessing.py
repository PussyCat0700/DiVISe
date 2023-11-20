import argparse
import logging
import os
import numpy as np
from scipy.io.wavfile import read
from tqdm import tqdm

LRS3_VALID_SUBDIRS = ["short-pretrain", "trainval", "test"]
logger = logging.Logger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lrs3-dir", help="preprocessed LRS3 dir(See AV-HuBERT on how to preprocess)")
    parser.add_argument("--pitch", choices=['kaldi', 'pyworld'])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    pitch_type = args.pitch
    if 'kaldi' == pitch_type:
        from kaldi.kaldi_pitch import pitch_single
        import torch
    elif 'pyworld' == pitch_type:
        from meldataset import pitch_single
    else:
        raise RuntimeError(f"{pitch_type=} unsupported")
    logger.info(pitch_single)
    if args.lrs3_dir:
        for split_dir in [os.path.join(args.lrs3_dir, "audio", x) for x in LRS3_VALID_SUBDIRS]:
            logger.info(f"Now processing {split_dir}")
            pbar = tqdm(os.walk(split_dir))
            for root, dirs, files in pbar:
                for file in files:
                    if file.endswith(".wav"):
                        wav_file_name = os.path.join(root, file)
                        pbar.set_description(f'now processing {wav_file_name}')
                        sr, wav_file = read(wav_file_name)
                        assert sr == 16000, f"{sr=} at {wav_file_name}"
                        pitch = pitch_single(wav_file, None, sampling_rate=sr)
                        serial_number = file.split('.')[0]
                        if 'pyworld' == pitch_type:
                            pitch_filename = f"{serial_number}_pw_dio.npy"
                            np.save(os.path.join(root, pitch_filename), pitch)
                        else:
                            pitch_filename = f"{serial_number}_kaldi.pt"
                            torch.save(pitch, os.path.join(root, pitch_filename))