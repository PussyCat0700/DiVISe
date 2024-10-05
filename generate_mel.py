import logging
import random
import sys
import warnings
import numpy as np
from tqdm import tqdm
from constants import GRIFFINLIM, HIFIGAN_NO_GRAD, HIFIGAN_WITH_GRAD, UNIT_HIFIGAN_NO_GRAD

from dataset import load_avhubert_config, load_dataset, get_dataloader
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import argparse
import json
import torch
import wandb
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict
from models import AVHuBERTGenerator
from utils import scan_checkpoint, load_checkpoint, seed_everything

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

# TODO magic path is bad
from pathlib import Path
se_path = Path("/data1/yfliu/model/CorentinJ/encoder.pt")


def generate_mel(rank, a, h, avhubert_config):
    if rank == 0 and a.wandb:
        full_path = os.path.abspath(a.checkpoint_path)
        pardir = os.path.abspath(f'{full_path}/{os.pardir}')
        proj_name = os.path.basename(pardir)
        run_name = os.path.basename(full_path)
        wandb.init(project=proj_name, name=run_name, sync_tensorboard=True)
        wandb.log({'processed_batch':0})
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    seed_everything(h.seed)
    torch.cuda.set_device(rank)  # A very strong boost. See https://github.com/jik876/hifi-gan/pull/25
    device = torch.device('cuda:{:d}'.format(rank))
    if a.train_mode == VIDEO2MEL_MODE:
        generator_mode = HIFIGAN_NO_GRAD if a.hifigan_ckpt is not None else GRIFFINLIM
        if h.unit_name is not None and h.unit_method == UNIT_HIFIGAN_NO_GRAD:
            generator_mode = UNIT_HIFIGAN_NO_GRAD
    elif a.train_mode == VIDEO2WAV_MODE:
        generator_mode = HIFIGAN_WITH_GRAD
    generator = AVHuBERTGenerator(hifigenerator_config=h,
                                  avhubert_model_config=avhubert_config["model"],
                                  generator_mode=generator_mode,
                                  svts=a.svts,
                                  ).to(device)

    if rank == 0:
        logging.info('model loaded.')
        os.makedirs(a.checkpoint_path, exist_ok=True)
        logging.info(f"checkpoints directory : {a.checkpoint_path}")

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')

    if a.avhubert_ckpt is not None:
        generator.load_pretrained_avhubertmodel(a.avhubert_ckpt, map_location=device)
    if a.svts:
        generator.load_spkencoder_for_svts(se_path, device=device)
    if a.hifigan_ckpt is not None:
        hifigan_weight = torch.load(a.hifigan_ckpt, map_location=device)
        def unwrap_module_generator(weight, ignore_conv_pre:bool):
            if ignore_conv_pre:
                return {'.'.join(k.split('.')[1:]):v for k,v in weight.items() if 'conv_pre' not in k}
            else:
                return {'.'.join(k.split('.')[1:]):v for k,v in weight.items()}
        generator.generator.load_state_dict(unwrap_module_generator(
            hifigan_weight["generator"]["model"], ignore_conv_pre=a.train_mode==VIDEO2WAV_MODE,
            ))
    # TODO: It is really unreasonable to keep all training states in state_dict_do, and it is still here just for compatibility.
    if a.train_mode == VIDEO2WAV_MODE:
        if cp_g is not None:
            state_dict_g = load_checkpoint(cp_g, device)
            generator.load_state_dict(state_dict_g['generator'])
    elif a.train_mode == VIDEO2MEL_MODE:
        if cp_g is None:
            state_dict_g = None
        else:
            state_dict_g = load_checkpoint(cp_g, device)
            generator.load_state_dict(state_dict_g['generator'])

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank], find_unused_parameters=True).to(device)

    kwargs = {
        "fake_km_mask":True,
    }
    avhubert_config["task"].max_sample_seconds = 10000 # Hacking: No Upper Limit
    trainset = load_dataset("train", avhubert_config["task"], **kwargs)
    validset = load_dataset("valid", avhubert_config["task"], **kwargs)
    all_datasets = trainset+validset
    data_loader, _ = get_dataloader(all_datasets, 
                                    batch_size=h.batch_size,  # for convenience of not dealing with masks
                                    num_workers=h.num_gpus if h.num_gpus>1 else 0, 
                                    dist_sampler=h.num_gpus > 1, 
                                    pin_memory=not h.num_gpus > 1,
                                    shuffle=False,
                                    drop_last=False,
                                    )

    # End of a train epoch
    generator.eval()
    torch.cuda.empty_cache()
        
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Generating MelSpectrogram...")
        generated_batches = 0
        for batch in pbar:
            avhubert_source_batch = batch["net_input"]["source"]
            mel_padding_mask = batch["net_input"]["padding_mask_mel"].to(device)
            skip=True
            for name in avhubert_source_batch["name"]:
                audio_base_dir = name["audio_basedir"]
                audio_id = name["audio_id"]
                mel_save_path = os.path.join(audio_base_dir, f'{audio_id}_mel_{a.postfix}.npy')
                if not os.path.exists(mel_save_path):
                    skip=False
                    break
            if skip:
                continue
            generator_out = generator(
                    avhubert_source_batch["video"].to(device), 
                    None,
                    ~mel_padding_mask,
                    avhubert_source_batch["audio"].to(device),  # for svts only
                    )
            y_g_avhubert_mels = generator_out["melspec_out"]
            lengths = (~mel_padding_mask).sum(dim=-1)
            y_g_avhubert_mels = y_g_avhubert_mels.transpose(1, 2).cpu().numpy()
            for y_g_avhubert_mel, name, length in zip(y_g_avhubert_mels, avhubert_source_batch["name"], lengths):
                audio_base_dir = name["audio_basedir"]
                audio_id = name["audio_id"]
                y_g_avhubert_mel = y_g_avhubert_mel[:length, :]
                mel_save_path = os.path.join(audio_base_dir, f'{audio_id}_mel_{a.postfix}.npy')
                np.save(mel_save_path, y_g_avhubert_mel)
            generated_batches += 1
            if rank == 0 and a.wandb:
                wandb.log({"processed_batch":generated_batches})
            pbar.set_description(f'{generated_batches=}')
            


def main():
    
    logging.info('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--hifigan_config', default='conf/hifigan/video2speech_template.json')  # TODO: Change back in formal release
    parser.add_argument('--avhubert_config', default='conf/avhubert/base_avhubert_30h.yaml')
    parser.add_argument('--avhubert_ckpt', help='if specified, will load pretrained weight onto AVHuBERTModel')
    parser.add_argument('--hifigan_ckpt', help='if specified, will load pretrained weight onto HiFi-GAN in v2w mode'\
        ' as part of the model or in v2m mode (with gradient) as mel-to-audio converter in v2w mode(without gradient)')
    parser.add_argument('--postfix', default='generated')
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--svts', action='store_true')
    parser.add_argument('--train_mode', choices=[VIDEO2MEL_MODE, VIDEO2WAV_MODE], default=VIDEO2MEL_MODE, help='v2w(video2wav), v2m(video2mel)')

    a = parser.parse_args()
    logging.info(f'Proceeding with mode {a.train_mode}')

    with open(a.hifigan_config) as f:
        data = f.read()

    json_config = json.loads(data)
    if 'unit_name' not in json_config:
        json_config['unit_name'] = None
    h = AttrDict(json_config)
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
    if h.num_gpus > 1:
        mp.spawn(generate_mel, nprocs=h.num_gpus, args=(a, h, avhubert_config))
    else:
        generate_mel(0, a, h, avhubert_config)


if __name__ == '__main__':
    main()
