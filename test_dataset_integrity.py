

import json
import logging
from tqdm import tqdm
from dataset.dataset_loading import get_dataloader, load_avhubert_config, load_dataset
from dataset.meldataset import mel_spectrogram_and_energy
from env import AttrDict
logging.getLogger(__name__)

if __name__ == '__main__':
    avhubert_config = load_avhubert_config("conf/avhubert/base_avhubert.yaml")
    with open("conf/hifigan/video2speech_v1.json") as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    sets = {}
    for split in ["train", "valid"]:
        sets[split] = {}
        sets[split]["dataset"] = load_dataset(split, avhubert_config["task"])
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
            assert src["pitch"].shape[-1] == y_mel.shape[-1], f'{y_mel.shape[-1]=} but {src["pitch"].shape[-1]=}'
            assert src["pitch"].shape[-1] == src["video"].shape[2] * 4, f'{src["video"].shape[2]=} but {src["pitch"].shape[-1]=}'
    logging.info("check successful")