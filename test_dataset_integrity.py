from ast import parse
import json
import logging
from tqdm import tqdm
from dataset.dataset_loading import get_dataloader, load_avhubert_config, load_dataset
from dataset.meldataset import mel_spectrogram_and_energy
import argparse
from env import AttrDict
logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--avhubert_config", default="conf/avhubert/base_avhubert_30h.yaml", help='replace with your config file')
    parser.add_argument("--hifigan_config", default="conf/hifigan/video2speech_template.json", help='config parts on dataset loading will not take effect in this script')
    parser.add_argument("--pitch_type")
    parser.add_argument("--km")
    parser.add_argument("--hu_name")
    args = parser.parse_args()
    avhubert_config = load_avhubert_config(args.avhubert_config)
    with open(args.hifigan_config) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    sets = {}
    for split in ["valid", "train"]:
        sets[split] = {}
        kwargs = {}
        if "train" == split:
            kwargs.update({
                "pitch_type":args.pitch_type,
                "km_name":args.km,
                "hu_name":args.hu_name
            })
        sets[split]["dataset"] = load_dataset(split, avhubert_config["task"], **kwargs)
        sets[split]["dataloader"], sets[split]["sampler"] = get_dataloader(sets[split]["dataset"], 
            batch_size=8,
            num_workers=0, 
            dist_sampler=False, 
            pin_memory=False,
            shuffle=False)
    for split in sets.keys():
        dataloader = sets[split]["dataloader"]
        pbar = tqdm(dataloader)
        for batch in pbar:
            src = batch["net_input"]["source"]
            y_dict = mel_spectrogram_and_energy(src["audio"], h.n_fft, h.num_mels,
                                  h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax,
                                  center=False)
            y_mel = y_dict["spec"]
            if src["pitch"] is not None:
                assert src["pitch"].shape[-1] == y_mel.shape[-1], f'{y_mel.shape[-1]=} but {src["pitch"].shape[-1]=}'
            if src["km"] is not None:
                assert src["km"].shape[-1] == src["video"].shape[2] * 2, f'{src["video"].shape[2]=} but {src["km"].shape[-1]=}'
            if src["hu"] is not None:
                assert src["hu"].shape[1] == src["video"].shape[2] * 2, f'{src["video"].shape[2]=} but {src["hu"].shape[1]=}'
    logging.info("check successful")