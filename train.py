import logging
import sys
import warnings
from omegaconf import OmegaConf

from tqdm import tqdm

from dataset import load_avhubert_config, load_dataset, get_dataloader
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import wandb
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from dataset.meldataset import mel_spectrogram, mel_spectrogram_and_energy, pitch
from models import AVHuBERTGenerator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

torch.backends.cudnn.benchmark = True
logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
logging.getLogger(__name__)

def train(rank, a, h, avhubert_config):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    torch.cuda.set_device(rank)  # A very strong boost. See https://github.com/jik876/hifi-gan/pull/25
    device = torch.device('cuda:{:d}'.format(rank))
    
    generator = AVHuBERTGenerator(hifigenerator_config=h,
                                  avhubert_model_config=avhubert_config["model"], 
                                  use_prosody=a.prosody,
                                  ).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if rank == 0:
        logging.info('model loaded.')
        os.makedirs(a.checkpoint_path, exist_ok=True)
        logging.info(f"checkpoints directory : {a.checkpoint_path}")

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if a.avhubert_ckpt is not None:
        generator.load_pretrained_avhubertmodel(a.avhubert_ckpt, map_location=device)
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)
    trainset = load_dataset("train", avhubert_config["task"])
    train_loader, train_sampler = get_dataloader(trainset, 
                                                batch_size=h.batch_size,
                                                num_workers=h.num_workers, 
                                                dist_sampler=h.num_gpus > 1,
                                                shuffle=True)

    if rank == 0:
        validset = load_dataset("valid", avhubert_config["task"])
        validation_loader, _ = get_dataloader(validset, 
                                            batch_size=h.batch_size,
                                            num_workers=h.num_workers, 
                                            dist_sampler=h.num_gpus > 1, 
                                            shuffle=False)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        train_ratio = epoch / a.training_epochs  # [0, 1-1/a.training_epochs]
        if rank == 0:
            start = time.time()
            logging.info("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)
        pbar = tqdm(train_loader)
        for batch in pbar:
            if rank == 0:
                start_b = time.time()
            # x, y, _, y_mel = batch
            """
            batch:
            # Useful for our training:
            id(1D Tensor): sample ids(index) from dataset
            net_input(dict): input for AV-HuBERT model
            utt_id(List): file paths of samples
            # Not useful for our training:
            target_lengths(1D Tensor): length of output of text label processor. It is not the label for our task.
            ntokens(int): total length of target_lengths.
            target(BxT Tensor): output of text label processor. It is not the target for our task.
            """
            avhubert_source_batch = batch["net_input"]["source"]
            y = avhubert_source_batch["audio"].to(device)
            wav_padding_mask = batch["net_input"]["padding_mask_wav"].to(device)
            mel_padding_mask = batch["net_input"]["padding_mask_mel"].to(device)
            y_dict = mel_spectrogram_and_energy(y, h.n_fft, h.num_mels,
                                  h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax,
                                  center=False)
            y_mel = y_dict["spec"]
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            generator_out = generator(avhubert_source_batch["video"].to(device))
            y_g_hat = generator_out["wav_generated"]
            y_g_avhubert_mel = generator_out["melspec_out"]
            if a.prosody:
                def normalize_prosody(x):
                    return (x - x.mean(dim=-1, keepdim=True))/x.std(dim=-1, keepdim=True)
                energy_targets = y_dict["energy"].to(device)
                pitch_targets = pitch(y, wav_padding_mask, h.sampling_rate, h.hop_size).to(device)
                energy_targets = normalize_prosody(energy_targets)
                pitch_targets = normalize_prosody(pitch_targets)
                pitch_predictions = generator_out["prosody"]["pitch_pred"]
                energy_predictions = generator_out["prosody"]["energy_pred"]
                
                pitch_loss = F.mse_loss(pitch_predictions.masked_select(~mel_padding_mask), pitch_targets.masked_select(~mel_padding_mask))
                energy_loss = F.mse_loss(energy_predictions.masked_select(~mel_padding_mask), energy_targets.masked_select(~mel_padding_mask))
            
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                          h.fmin, h.fmax_for_loss)

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel.masked_select(~mel_padding_mask.unsqueeze(1)), y_g_hat_mel.masked_select(~mel_padding_mask.unsqueeze(1))) * 45
            # Another L1 Mel-Spectrogram Loss from AV-HuBERT Generator itself.
            alpha_avhubert = 0 if train_ratio>1/5 else -5*train_ratio+1  # 1.0 if ratio==0, 0.0 if ratio==1/5
            alpha_avhubert = h.base_alpha_avhubert*alpha_avhubert
            loss_mel_avhubert = F.l1_loss(y_mel.masked_select(~mel_padding_mask.unsqueeze(1)), y_g_avhubert_mel.masked_select(~mel_padding_mask.unsqueeze(1))) * alpha_avhubert

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel + loss_mel_avhubert
            if a.prosody:
                loss_gen_all += pitch_loss + energy_loss
            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error_generator = F.l1_loss(y_mel, y_g_hat_mel).item()
                        mel_error_avhubert = F.l1_loss(y_mel, y_g_avhubert_mel).item()

                    pbar.set_description('Epoch: {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(epoch, loss_gen_all, mel_error_generator, time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, 
                                    {'mpd': (mpd.module if h.num_gpus > 1
                                                         else mpd).state_dict(),
                                     'msd': (msd.module if h.num_gpus > 1
                                                         else msd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    def log_training(tag, value):
                        sw.add_scalar(f"training/{tag}", value, steps)
                    log_training("gen_loss_total", loss_gen_all)
                    log_training("mel_spec_error_generator", mel_error_generator)
                    log_training("mel_spec_error_avhubert", mel_error_avhubert)
                    if a.prosody:
                        log_training("pitch_regression_mse", pitch_loss)
                        log_training("energy_regression_mse", energy_loss)
                    log_training("epoch", epoch)
                    log_training("alpha_avhubert", alpha_avhubert)

                # Validation
                if steps % a.validation_interval == 0 and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = {
                        "mel_spec_error_generator": 0,
                        "mel_spec_error_avhubert": 0,
                    }
                    with torch.no_grad():
                        pbar2 = tqdm(validation_loader, desc="Validation in progress...")
                        for j, batch in enumerate(pbar2):
                            avhubert_source_batch = batch["net_input"]["source"]
                            y = avhubert_source_batch["audio"].to(device)
                            y_mel = mel_spectrogram(y, h.n_fft, h.num_mels,
                                                h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax,
                                                center=False)
                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                            generator_out = generator(avhubert_source_batch["video"].to(device))
                            y_g_hat = generator_out["wav_generated"]
                            y_g_avhubert_mel = generator_out["melspec_out"]
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          h.hop_size, h.win_size,
                                                          h.fmin, h.fmax_for_loss)
                            val_err_tot["mel_spec_error_generator"] += F.l1_loss(y_mel, y_g_hat_mel).item()
                            val_err_tot["mel_spec_error_avhubert"] += F.l1_loss(y_mel, y_g_avhubert_mel).item()

                            if j <= 4:
                                text = batch["target"]
                                if steps // a.validation_interval == 1:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    sw.add_text('gt/y_text_{}'.format(j), text[0], steps)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(y_mel[0].cpu()), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(y_g_hat[0].cpu(), h.n_fft, h.num_mels,
                                                             h.sampling_rate, h.hop_size, h.win_size,
                                                             h.fmin, h.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                        for val_err_key, val_err_term in val_err_tot.items():
                            val_err = val_err_term / (j+1)
                            sw.add_scalar(f"validation/{val_err_key}", val_err, steps)

                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            logging.info('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    
    logging.info('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    default_ckpt_dir = 'cp_hifigan'
    parser.add_argument('--checkpoint_path', default=default_ckpt_dir)
    parser.add_argument('--hifigan_config', default='conf/hifigan/video2speech_v1.json')
    parser.add_argument('--avhubert_config', default='conf/avhubert/base_avhubert.yaml')
    parser.add_argument('--avhubert_ckpt', help='if specified, will load pretrained weight onto AVHuBERTModel')
    parser.add_argument('--training_epochs', default=100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=37383, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=37383, type=int)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--no_prosody', action='store_true')

    a = parser.parse_args()
    a.prosody = not a.no_prosody
    if a.checkpoint_path == default_ckpt_dir:
        logging.warning(f"You're using default checkpoint dir {default_ckpt_dir}.\n"+\
            " This should not happen in serious runs as checkpoint dir is likely overwritten with runs in default args.")

    with open(a.hifigan_config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.hifigan_config, 'hifigan_config.json', a.checkpoint_path)
    
    avhubert_config = load_avhubert_config(a.avhubert_config)
    OmegaConf.save(avhubert_config, os.path.join(a.checkpoint_path, 'avhubert_config.yaml'))

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        logging.info(f'Batch size per GPU :{h.batch_size}')
    else:
        pass
    if a.wandb:
        proj_name = os.path.basename(a.checkpoint_path)
        wandb.init(project=proj_name, sync_tensorboard=True)
    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h, avhubert_config))
    else:
        train(0, a, h, avhubert_config)


if __name__ == '__main__':
    main()
