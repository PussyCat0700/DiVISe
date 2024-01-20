import logging
import random
import sys
import warnings
import editdistance
import torchaudio
from tqdm import tqdm
from audio.eval_utils import GreedyCTCDecoder

from dataset import load_avhubert_config, load_dataset, get_dataloader
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import argparse
import json
import torch
import wandb
import torch.multiprocessing as mp
from env import AttrDict
from utils import seed_everything

torch.backends.cudnn.benchmark = True
logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
logging.getLogger(__name__)

VIDEO2MEL_MODE = "v2m"
VIDEO2WAV_MODE = "v2w"

def eval_asr(rank, a, h, avhubert_config):
    if rank == 0 and a.wandb:
        wandb.init(project='ASRtest', name=a.run_name, sync_tensorboard=True)

    seed_everything(h.seed)
    torch.cuda.set_device(rank)  # A very strong boost. See https://github.com/jik876/hifi-gan/pull/25
    device = torch.device('cuda:{:d}'.format(rank))

    dataloading_kwargs = {}
    if h.unit_name is not None:
        dataloading_kwargs = {
            "km_pad_class_idx": h.k,
        }

    kwargs = {
        "fake_km_mask":True,
    }
    kwargs.update(**dataloading_kwargs)
    avhubert_config["task"].max_sample_seconds = 10000 # Hacking: No Upper Limit
    all_datasets = load_dataset("test", avhubert_config["task"], **kwargs)
    data_loader, _ = get_dataloader(all_datasets, 
                                    batch_size=h.batch_size,  # for convenience of not dealing with masks
                                    num_workers=h.num_gpus if h.num_gpus>1 else 0, 
                                    dist_sampler=h.num_gpus > 1, 
                                    pin_memory=not h.num_gpus > 1,
                                    shuffle=False,
                                    drop_last=False,
                                    )
    err_tot = {"wer_average":0.0, "wer_algorithmic":0.0}
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    torch.cuda.empty_cache()
        
    with torch.no_grad():
        transcriber = bundle.get_model().to(device)
        valid_greedy_decoder = GreedyCTCDecoder(labels=bundle.get_labels())
        pbar = tqdm(data_loader, desc="Evaluating ASR...")
        n_err, n_total = 0, 0
        for j, batch in enumerate(pbar):
            avhubert_source_batch = batch["net_input"]["source"]
            y = avhubert_source_batch["audio"].to(device)
            gt_texts = [x.strip() for x in batch["target"]]
            wav_padding_mask = batch["net_input"]["padding_mask_wav"].to(device)
            lengths = (~wav_padding_mask).sum(dim=-1)
            # model definition can be found in https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/model.html
            emissions, lengths = transcriber(y, lengths)  # length indicates the valid length in time axis of emissions
            # reference for WER calculation: https://github.com/facebookresearch/av_hubert/blob/258fb50e155134eec2c4b49c2ae8de267075fd18/avhubert/infer_s2s.py#L254
            for emission, gt_text, length in zip(emissions, gt_texts, lengths):
                generated_text = valid_greedy_decoder(emission, length)
                hypo, ref = generated_text.strip().split(), gt_text.strip().split()
                n_err += editdistance.eval(hypo, ref)
                n_total += len(ref)
            err_tot["wer_algorithmic"] = n_err/n_total
            pbar.set_description(f'wer={err_tot["wer_average"]/(j+1)} and new wer={err_tot["wer_algorithmic"]}')
        if rank == 0 and a.wandb:
            wandb.log({"wer_final":err_tot["wer_algorithmic"]})
            


def main():
    
    logging.info('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--hifigan_config', default='conf/hifigan/video2speech_template.json')  # TODO: Change back in formal release
    parser.add_argument('--avhubert_config', default='conf/avhubert/base_avhubert_30h.yaml')
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--run_name', help='if specified then wandb will be used')
    parser.add_argument('--predicted-prosody', action='store_true', help='(deprecated) if specified, will use predicted prosody instead of GT in training.')
    parser.add_argument('--train_mode', choices=[VIDEO2MEL_MODE, VIDEO2WAV_MODE], default=VIDEO2MEL_MODE, help='v2w(video2wav), v2m(video2mel)')

    a = parser.parse_args()
    a.real_prosody = not a.predicted_prosody
    a.wandb = a.run_name is not None
    logging.info(f'Proceeding with mode {a.train_mode}')

    with open(a.hifigan_config) as f:
        data = f.read()

    json_config = json.loads(data)
    if 'prosody_type' not in json_config:
        json_config['prosody_type'] = None
    if 'unit_name' not in json_config:
        json_config['unit_name'] = None
    if 'hu_repr_name' not in json_config:
        json_config['hu_repr_name'] = None
    h = AttrDict(json_config)
    if h.prosody_type is not None:
        assert h.norm_mode in ['original', 'meanvar'], f"{h.norm_mode=} which is not a valid way to normalize prosody."
    if a.train_mode == VIDEO2MEL_MODE:
        # give random port to avoid collision
        url = h.dist_config['dist_url']
        splits = url.split(":")
        port = int(splits[-1])
        port -= random.randint(100, 1000)
        h.dist_config['dist_url'] = ':'.join(splits[:-1]+[str(port)])
    # build_env(a.hifigan_config, 'hifigan_config.json', a.checkpoint_path)
    
    avhubert_config = load_avhubert_config(a.avhubert_config)
    # OmegaConf.save(avhubert_config, os.path.join(a.checkpoint_path, 'avhubert_config.yaml'))

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        logging.info(f'running with {h.num_gpus} GPUs and batch size of {h.batch_size}')
    else:
        pass
    eval_asr(0, a, h, avhubert_config)


if __name__ == '__main__':
    main()
