from datetime import timedelta
import math
import logging
import random
import sys
import warnings
from omegaconf import OmegaConf
from transformers import Wav2Vec2ForCTC
from tqdm import tqdm
from constants import GRIFFINLIM,\
    GENERATOR_METHODS, HIFIGAN_NO_GRAD, HIFIGAN_WITH_GRAD, BIGVGAN_NO_GRAD, PWG_NO_GRAD,\
    UNIT_METHODS, UNIT_SPEECH_TOKENIZER_NO_GRAD, \
    VALID_MODE, TEST_MODE

from contrastive.metrics import EERMetric
from dataset import load_avhubert_config, load_hifigan_config, load_dataset, get_dataloader
from dataset.dataset_loading import load_dataset_eer
import speaker_encoder.inference as corentinJEncoder
from vocoders.parallel_wavegan.pwg_model import PWGModel
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
from dataset.meldataset import MelSpectrogramInverter,LogMelSpectrogram
from models import AVHuBERTGenerator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss
from utils import DataLoaderSeeder, TriStageLRScheduler, plot_spectrogram, save_wav_16khz, scan_checkpoint, load_checkpoint, save_checkpoint, seed_everything, unwrap_module_discriminator, unwrap_module_generator
from audio.eval_utils import MetricsEvaluater, MyWav2Vec2Processor, SampleSaver
import light_hf_proxy


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
metrics = {}
best_metrics = None
steps = 0


# TODO magic path is bad
from pathlib import Path
se_path = Path("/data1/yfliu/model/CorentinJ/encoder.pt")


def train(rank, a, h, avhubert_config):
    global metrics, best_metrics, steps
    if rank == 0 and a.wandb:
        full_path = os.path.abspath(a.checkpoint_path)
        pardir = os.path.abspath(f'{full_path}/{os.pardir}')
        proj_name = os.path.basename(pardir)
        run_name = os.path.basename(full_path)
        if a.test:
            run_name += '-test'
            if a.hifigan_ckpt is not None:
                run_name += '_hifigan'
            elif a.bigvgan_ckpt is not None:
                run_name += '_bigvgan'
            elif a.pwg_ckpt is not None:
                run_name += '_pwg'
        wandb.init(project=proj_name, name=run_name, sync_tensorboard=True)
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank, timeout=timedelta(seconds=7200000),)

    seed_everything(h.seed)
    torch.cuda.set_device(rank)  # A very strong boost. See https://github.com/jik876/hifi-gan/pull/25
    device = torch.device('cuda:{:d}'.format(rank))
    logmel = LogMelSpectrogram().to(device)
    if a.train_mode == VIDEO2MEL_MODE:
        generator_mode = GRIFFINLIM
        if a.hifigan_ckpt is not None:
            generator_mode = HIFIGAN_NO_GRAD
        if a.bigvgan_ckpt is not None:
            generator_mode = BIGVGAN_NO_GRAD
        if a.pwg_ckpt is not None:
            generator_mode = PWG_NO_GRAD
        if h.unit_name is not None:
            generator_mode = h.unit_method
    elif a.train_mode == VIDEO2WAV_MODE:
        generator_mode = HIFIGAN_WITH_GRAD
    unit_key = None
    if h.unit_name is not None:
        speech_tokenizer_enabled = h.unit_method in [UNIT_SPEECH_TOKENIZER_NO_GRAD]
        if speech_tokenizer_enabled:
            unit_key = "st"
        else:
            unit_key = "km"
    use_farl = a.use_farl
    use_svts = a.svts
    generator = AVHuBERTGenerator(hifigenerator_config=h,
                                  avhubert_model_config=avhubert_config["model"], 
                                  generator_mode=generator_mode,
                                  use_farl=use_farl,
                                  svts=use_svts,
                                  ).to(device)
    generator_module = generator
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if rank == 0:
        logging.info('model loaded.')
        os.makedirs(a.checkpoint_path, exist_ok=True)
        logging.info(f"checkpoints directory : {a.checkpoint_path}")

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_', load_best=a.test)
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_', load_best=a.test)

    if a.avhubert_ckpt is not None:
        generator.load_pretrained_avhubertmodel(a.avhubert_ckpt, map_location=device)
    if use_svts:
        generator.load_spkencoder_for_svts(se_path, device=device)
    if a.hifigan_ckpt is not None and a.train_mode == VIDEO2WAV_MODE:
        # hifigan ckpt might be updated in training with 2wav mode
        hifigan_weight = torch.load(a.hifigan_ckpt, map_location=device)
        generator.generator.load_state_dict(unwrap_module_generator(
            hifigan_weight["generator"]["model"], ignore_conv_pre=a.train_mode==VIDEO2WAV_MODE,
            ))
        mpd.load_state_dict(unwrap_module_discriminator(hifigan_weight["discriminator"]["model"], "mpd"))
        msd.load_state_dict(unwrap_module_discriminator(hifigan_weight["discriminator"]["model"], "msd"))
    if a.train_mode == VIDEO2WAV_MODE:
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
            best_metrics = state_dict_do['metrics']
    elif a.train_mode == VIDEO2MEL_MODE:
        if cp_g is None:
            state_dict_g = None
            last_epoch = -1
        else:
            state_dict_g = load_checkpoint(cp_g, device)
            # TODO this is redundant but works like a trap. Careful if you want to lint them
            if h.unit_name is not None and h.unit_method in UNIT_METHODS:
                generator.load_state_dict(state_dict_g['generator'], strict=not a.skip_ckptcheck)
            else:
                generator.load_full_model_weight(state_dict_g['generator'], ignore_generator=True)
            steps = state_dict_g['steps'] + 1
            last_epoch = state_dict_g['epoch']
            best_metrics = state_dict_g['metrics']
    if a.hifigan_ckpt is not None:
        # hifigan ckpt is not supposed to be updated in training with 2mel mode
        hifigan_weight = torch.load(a.hifigan_ckpt, map_location=device)
        generator.generator.load_state_dict(unwrap_module_generator(
            hifigan_weight["generator"]["model"], ignore_conv_pre=a.train_mode==VIDEO2WAV_MODE,
            ))
    if a.bigvgan_ckpt is not None:
        bigvgan_weight = torch.load(a.bigvgan_ckpt, map_location=device)
        generator.generator.load_state_dict(bigvgan_weight["generator"])
    if a.pwg_ckpt is not None:
        generator.generator = PWGModel(a.pwg_ckpt).to(device)
        generator.generator.eval()
        for param in generator.generator.parameters():
            param.requires_grad = False

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank], find_unused_parameters=True).to(device)
        generator_module = generator.module
        if a.train_mode == VIDEO2WAV_MODE:
            mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
            msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)
    optim_g = torch.optim.AdamW([p for p in generator.parameters() if p.requires_grad], h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    if a.train_mode == VIDEO2WAV_MODE:
        optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                    h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    actual_total_updates = math.ceil(h.total_updates / h.num_gpus)
    part_updates = math.ceil(actual_total_updates / a.n_ckpts)
    saving_updates = {x for x in range(part_updates, actual_total_updates+1, part_updates)}
    logging.info(f"{actual_total_updates=}")
    if not a.test:
        if a.train_mode == VIDEO2WAV_MODE:
            if state_dict_do is not None:
                optim_g.load_state_dict(state_dict_do['optim_g'])
                optim_d.load_state_dict(state_dict_do['optim_d'])
        elif a.train_mode == VIDEO2MEL_MODE:
            if state_dict_g is not None:
                optim_g.load_state_dict(state_dict_g['optim_g'])
        if not h.revise_setting:
            scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
        else:
            actual_frozen_updates = math.ceil(h.frozen_steps / h.num_gpus)
            if steps <= actual_frozen_updates:
                logging.info(f"AVHuBERT will be frozen for {actual_frozen_updates} updates.")
                generator_module.frontend_with_encoder.avhubert_grad(False)
            else:
                logging.info(f"current {steps=}. AVHuBERT will not be frozen after {actual_frozen_updates} updates.")
            # Exactly as in ReVISE Tab. 17
            scheduler_g = TriStageLRScheduler(optim_g, actual_total_updates, h.t1_percent, h.t2_percent, last_lr_factor=h.last_lr_factor,last_epoch=steps-1)
        if a.train_mode == VIDEO2WAV_MODE:
            scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    dataloading_kwargs = {}
    if h.unit_name is not None:
        dataloading_kwargs = {
            "km_name":h.unit_name,
            "st_type":h.st_type,
            "km_pad_class_idx": h.k,
            "generator_mode":generator_mode,
            "with_image_tsv": use_farl,
        }
    trainset = load_dataset("train", avhubert_config["task"], **dataloading_kwargs)
    train_loader, train_sampler = get_dataloader(trainset, 
                                                batch_size=h.batch_size,
                                                num_workers=h.num_gpus, 
                                                dist_sampler=h.num_gpus > 1,
                                                pin_memory=not h.num_gpus > 1,
                                                shuffle=True,
                                                seeder=DataLoaderSeeder(h.seed),
                                                )

    sw = None
    if rank == 0:
        kwargs = {
            "st_type": h.st_type
        }
        if h.unit_name is not None and h.valid_unit_name is not None:
            # You can apply trained kmeans model on valid set to get km labels just for reference.
            kwargs.update({
                "km_name":h.valid_unit_name,
                })
        else:
            kwargs.update({
                "fake_km_mask":True,
            })
        dataloading_kwargs.update(**kwargs)
        validset = load_dataset("valid", avhubert_config["task"], **dataloading_kwargs)
        validation_loader, _ = get_dataloader(validset, 
                                            batch_size=h.batch_size,
                                            num_workers=1, 
                                            drop_last=False,
                                            shuffle=False)
        if h.unit_name is not None and h.test_unit_name is not None:
            # You can apply trained kmeans model on valid set to get km labels just for reference.
            kwargs.update({
                "km_name":h.test_unit_name,
                })
        else:
            kwargs.update({
                "fake_km_mask":True,
            })
        avhubert_config["task"].max_sample_seconds = 10000 # Hacking: No Upper Limit
        dataloading_kwargs.update(**kwargs)
        testset = load_dataset("test", avhubert_config["task"], **dataloading_kwargs)
        test_loader, _ = get_dataloader(testset, 
                                        batch_size=h.batch_size,
                                        num_workers=1, 
                                        drop_last=False,
                                        shuffle=False)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    if a.train_mode == VIDEO2WAV_MODE:
        mpd.train()
        msd.train()
    # Griffin-Lim is used in comparison with Vocoder
    mel2wav_inverter = MelSpectrogramInverter(h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax, device)
    mel2wav_inverter.eval()
    a.training_epochs = math.ceil(h.total_updates / h.num_gpus / len(train_loader))
    logging.info(f"{a.training_epochs=}")
    w2v_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(device)
    w2v_processor = MyWav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    if not a.test:
        for epoch in range(max(0, last_epoch), a.training_epochs):
            train_ratio = epoch / a.training_epochs  # [0, 1-1/a.training_epochs]
            generator.train()
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
                image_input = avhubert_source_batch["images"]
                if image_input is not None:
                    image_input = image_input.to(device)
                mel_padding_mask = batch["net_input"]["padding_mask_mel"].to(device)
                wav_padding_mask = batch["net_input"]["padding_mask_wav"].to(device)
                y_mel = logmel(y)
                y = torch.autograd.Variable(y.to(device, non_blocking=True))
                y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                y = y.unsqueeze(1)
                unit_target = {
                    "kmeans_target":None,
                    "kmeans_mask":None,
                }
                padding_mask = ~mel_padding_mask
                if h.unit_name is not None:
                    kmeans_mask = batch["net_input"]["padding_mask_km"].to(device)
                    padding_mask = ~kmeans_mask
                    if h.unit_name is not None:
                        unit_target["kmeans_target"] = avhubert_source_batch[unit_key].to(device)
                        unit_target["kmeans_mask"] = ~kmeans_mask

                generator_out = generator(
                    avhubert_source_batch["video"].to(device), 
                    image_input,
                    padding_mask,
                    y,  # for svts only
                    )
                y_g_avhubert_mel = generator_out["melspec_out"]
                if h.unit_name is not None:
                    kmeans_targets = unit_target["kmeans_target"]         
                    if h.unit_method in GENERATOR_METHODS:
                        if h.unit_method == HIFIGAN_NO_GRAD:
                            unit_predictions = generator_out["unit"]["kmeans_pred"]
                        elif h.unit_method in UNIT_METHODS:
                            unit_predictions = generator_out["revise_logits"]
                        C = unit_predictions.shape[-1]
                        unit_predictions = unit_predictions.masked_select((~kmeans_mask).unsqueeze(-1)).reshape(-1, C)
                        kmeans_targets = kmeans_targets.masked_select(~kmeans_mask)
                        unit_loss = F.cross_entropy(unit_predictions, kmeans_targets)
                        
                    unit_loss = h.unit_scale*unit_loss
                    
                if a.train_mode == VIDEO2WAV_MODE:
                    y_g_hat = generator_out["wav_generated"]
                    y_g_hat_mel = logmel(y_g_hat.squeeze(1))

                    optim_d.zero_grad()

                    # TODO: Add mask for these GAN losses, which is however absent in HiFi-GAN's original setting?
                    # MPD
                    y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
                    loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g, ~wav_padding_mask, 'mpd')

                    # MSD
                    y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
                    loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g, ~wav_padding_mask, 'msd')

                    loss_disc_all = loss_disc_s + loss_disc_f

                    loss_disc_all.backward()
                    optim_d.step()

                # Generator
                optim_g.zero_grad()
                loss_gen_all = 0
                if a.train_mode == VIDEO2WAV_MODE:
                    # L1 Mel-Spectrogram Loss
                    loss_mel = F.l1_loss(y_mel.masked_select(~mel_padding_mask.unsqueeze(1)), y_g_hat_mel.masked_select(~mel_padding_mask.unsqueeze(1))) * 45
                    loss_gen_all += loss_mel
                # Another L1 Mel-Spectrogram Loss from AV-HuBERT Generator itself.
                if a.decay_melloss:
                    alpha_avhubert = 0 if train_ratio>1/5 else -5*train_ratio+1  # 1.0 if ratio==0, 0.0 if ratio==1/5
                    alpha_avhubert = h.base_alpha_avhubert*alpha_avhubert
                else:
                    alpha_avhubert = h.base_alpha_avhubert
                if y_g_avhubert_mel is not None:
                    loss_mel_avhubert = F.l1_loss(y_mel.masked_select(~mel_padding_mask.unsqueeze(1)), y_g_avhubert_mel.masked_select(~mel_padding_mask.unsqueeze(1))) * alpha_avhubert

                if a.train_mode == VIDEO2WAV_MODE:
                    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
                    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
                    # TODO: If masked loss is nice, apply it on feature_loss (which will be troublesome work too).
                    loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                    loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                    loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g, ~wav_padding_mask, 'mpd')
                    loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g, ~wav_padding_mask, 'msd')
                    loss_gen_all = loss_gen_all + loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f
                # Used to be loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel + loss_mel_avhubert
                if y_g_avhubert_mel is not None:
                    # ReVISE doesn't need this loss
                    loss_gen_all += loss_mel_avhubert
                if h.unit_name is not None:
                    loss_gen_all += unit_loss
                loss_gen_all.backward()
                optim_g.step()

                if rank == 0:
                    # STDOUT logging
                    if steps % a.stdout_interval == 0:
                        with torch.no_grad():
                            if a.train_mode == VIDEO2WAV_MODE:
                                mel_error_generator = F.l1_loss(y_mel.masked_select(~mel_padding_mask.unsqueeze(1)), y_g_hat_mel.masked_select(~mel_padding_mask.unsqueeze(1))).item()
                            if y_g_avhubert_mel is not None:
                                mel_error_avhubert = F.l1_loss(y_mel.masked_select(~mel_padding_mask.unsqueeze(1)), y_g_avhubert_mel.masked_select(~mel_padding_mask.unsqueeze(1))).item()
                        if a.train_mode == VIDEO2WAV_MODE:
                            pbar.set_description('Epoch: {:d}, Gen Loss Total : {:4.3f}, Video2Wav Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                                format(epoch, loss_gen_all, mel_error_generator, time.time() - start_b))
                        elif a.train_mode == VIDEO2MEL_MODE:
                            if y_g_avhubert_mel is not None:
                                pbar.set_description('Epoch: {:d}, Gen Loss Total : {:4.3f}, Video2Mel Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                                    format(epoch, loss_gen_all, mel_error_avhubert, time.time() - start_b))
                            else:
                                # ReVISE logging
                                pbar.set_description('Epoch: {:d}, Gen Loss Total : {:4.3f}, Unit. Error : {:4.3f}, s/b : {:4.3f}'.
                                    format(epoch, loss_gen_all, unit_loss, time.time() - start_b))

                    # Tensorboard summary logging
                    if steps % a.summary_interval == 0:
                        def log_training(tag, value):
                            sw.add_scalar(f"training/{tag}", value, steps)
                        log_training("gen_loss_total", loss_gen_all)
                        if a.train_mode == VIDEO2WAV_MODE:
                            log_training("mel_spec_error_generator", mel_error_generator)
                        if y_g_avhubert_mel is not None:
                            log_training("mel_spec_error_avhubert", mel_error_avhubert)
                        if h.unit_name is not None:
                            log_training("unit_loss", unit_loss)
                        log_training("epoch", epoch)
                        log_training("alpha_avhubert", alpha_avhubert)

                steps += 1
                if h.revise_setting:
                    # scheduler is updated step-level
                    scheduler_g.step()
                    if steps == actual_frozen_updates:
                        generator_module.frontend_with_encoder.avhubert_grad(True)
                        if rank == 0:
                            sw.add_scalar(f"training/unfreeze_step", steps, steps)
                # Validation&Checkpointing
                if steps in saving_updates and rank == 0:
                    val_args = {
                        "generator":generator.module if h.num_gpus > 1 else generator,
                        "w2v_processor":w2v_processor,
                        "w2v_model":w2v_model,
                        "a":a,
                        "h":h,
                        "device":device,
                        "loader":validation_loader,
                        "epoch":epoch,
                        "mel2wav_inverter":mel2wav_inverter,
                        "mode":VALID_MODE,
                        "sw":sw,
                        "unit_key":unit_key,
                    }
                    validate(**val_args)
                    # checkpointing
                    # TODO: It is unreasonable to put training states in discriminator checkpoints but due to inherent design we keep it here.
                    def save_all_checkpoints(save_title, remove_title=None):
                        checkpoint_path = "{}/g_{}".format(a.checkpoint_path, save_title)
                        prev_checkpoint_path_g = "{}/g_{}".format(a.checkpoint_path, remove_title) if remove_title is not None else None
                        if a.train_mode == VIDEO2WAV_MODE:
                            save_checkpoint(checkpoint_path,
                                            {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()},
                                            )
                            checkpoint_path = "{}/do_{}".format(a.checkpoint_path, save_title)
                            prev_checkpoint_path_do = "{}/do_{}".format(a.checkpoint_path, remove_title) if remove_title is not None else None
                            save_checkpoint(checkpoint_path, 
                                            {'mpd': (mpd.module if h.num_gpus > 1
                                                                else mpd).state_dict(),
                                            'msd': (msd.module if h.num_gpus > 1
                                                                else msd).state_dict(),
                                            'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                            'epoch': epoch, 'metrics': metrics,},
                                            )
                        elif a.train_mode == VIDEO2MEL_MODE:
                            # do_{} is not saved and every training state is kept in g_{}
                            save_checkpoint(checkpoint_path,
                                            {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict(),
                                            'optim_g': optim_g.state_dict(), 'steps': steps,
                                            'epoch': epoch, 'metrics': metrics,},
                                            )
                            prev_checkpoint_path_do = None
                        for filepath_to_remove in [prev_checkpoint_path_do, prev_checkpoint_path_g]:
                            if filepath_to_remove:
                                if os.path.exists(filepath_to_remove):
                                    os.remove(filepath_to_remove)
                                    logging.info(f'removed {filepath_to_remove}')
                                else:
                                    logging.warning(f'{filepath_to_remove} does not exist and removing is cancelled.')
                    save_all_checkpoints(steps, remove_title=steps-part_updates if steps-part_updates>0 else None)
                    if (h.lower_the_better and metrics[h.save_on_metric] <= best_metrics[h.save_on_metric]) \
                        or (not h.lower_the_better and metrics[h.save_on_metric] >= best_metrics[h.save_on_metric]):
                        save_all_checkpoints("best")
                        best_metrics = metrics
                if steps == actual_total_updates:
                    # Early breaking
                    break
            if not h.revise_setting:
                scheduler_g.step()
            if a.train_mode == VIDEO2WAV_MODE:  
                scheduler_d.step()
            
            if rank == 0:
                logging.info('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))
            # End of a train epoch
    if rank == 0:
        print('testing wer and so on.')
        tmpdir = None
        if a.save_samples:
            tmpdir = os.path.join(a.checkpoint_path, 'test_samples')
            os.makedirs(tmpdir, exist_ok=True)
        test_args = {
                "generator":generator.module if h.num_gpus > 1 else generator,
                "w2v_processor":w2v_processor,
                "w2v_model":w2v_model,
                "a":a,
                "h":h,
                "device":device,
                "loader":test_loader,
                "epoch":0,
                "mel2wav_inverter":mel2wav_inverter,
                "mode":TEST_MODE,
                "sw":sw,
                "unit_key":unit_key,
                "tmpdir":tmpdir,
            }
        validate(**test_args)
    if a.test_all:
        # Ultimate test. Time consuming
        print('testing eer.')
        test_eer(
            model=generator,
            device=device,
            args=a,
            h=h,
            sw=sw,
        )


def validate(
    generator:AVHuBERTGenerator,
    w2v_processor,
    w2v_model,
    a,
    h,
    device,
    loader,
    epoch,
    unit_key,
    mel2wav_inverter:MelSpectrogramInverter=None,
    mode=VALID_MODE,
    sw:SummaryWriter=None,
    tmpdir:str=None,
    ):
    global metrics, best_metrics, steps
    generator.eval()
    logmel = LogMelSpectrogram().to(device)
    err_tot = {"mel_spec_error_avhubert": 0}
    calc_classfication_metrics = h.unit_name is not None
    num_classes = None
    torch.cuda.empty_cache()
    if mode == VALID_MODE:
        gt_prefix = "gt"
        generated_prefix = "generated"
    elif mode == TEST_MODE:
        gt_prefix = "gt_test"
        generated_prefix = f"generated_test"
    f_gt = open(os.path.join(a.checkpoint_path, f'{gt_prefix}.txt'), 'w+')
    f_gf = open(os.path.join(a.checkpoint_path, f'{generated_prefix}_gf.txt'), 'w+')
    f_vc = open(os.path.join(a.checkpoint_path, f'{generated_prefix}_vc.txt'), 'w+')   
    if tmpdir is not None:
        samplesaver = SampleSaver(tmpdir)
    if calc_classfication_metrics:
        num_classes = h.k
        if generator.with_extra_padding_unit:
            num_classes += 1
    with torch.no_grad():
        audioeval_gf = MetricsEvaluater(
            w2v_processor=w2v_processor, 
            w2v_model=w2v_model,
            device=device,
            )
        audioeval_vocoder = MetricsEvaluater(
            w2v_processor=w2v_processor, 
            w2v_model=w2v_model,
            device=device,
            postfix="vocoder",
            num_classes=num_classes,
            )
        pbar = tqdm(loader, desc="Validation in progress...")
        for j, batch in enumerate(pbar):
            avhubert_source_batch = batch["net_input"]["source"]
            y = avhubert_source_batch["audio"].to(device)
            image_input = avhubert_source_batch["images"]
            if image_input is not None:
                image_input = image_input.to(device)
            gt_texts = [x.strip() for x in batch["target"]]
            mel_padding_mask = batch["net_input"]["padding_mask_mel"].to(device)
            wav_padding_mask = batch["net_input"]["padding_mask_wav"].to(device)
            y_mel = logmel(y)
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            unit_target = {
                "kmeans_target":None,
                "kmeans_mask":None,
            }
            padding_mask = ~mel_padding_mask
            if h.unit_name is not None:
                kmeans_mask = batch["net_input"]["padding_mask_km"]
                padding_mask = None
                if kmeans_mask is not None:
                    kmeans_mask = kmeans_mask.to(device)
                    unit_target["kmeans_mask"] = ~kmeans_mask
                    padding_mask = ~kmeans_mask
            generator_out = generator(avhubert_source_batch["video"].to(device), 
                                      image_input,
                                      padding_mask,
                                      y,  # for svts only
                                      )
            y_g_avhubert_mel = generator_out["melspec_out"]
            y_g_hat = None
            y_g_hat_vc = None
            if y_g_avhubert_mel is not None:
                y_g_hat = mel2wav_inverter(y_g_avhubert_mel.detach().transpose(-1, -2))
            if generator.with_generator:
                y_g_hat_vc = generator_out["wav_generated"]
            if tmpdir:
                # AV sync export
                saved_samples = samplesaver(wav_padding_mask, avhubert_source_batch["name"], y_g_hat_vc, y_g_hat)
            preds_km = None
            targets_km = None
            if h.unit_name is not None and avhubert_source_batch[unit_key] is not None:
                with torch.inference_mode():
                    if h.unit_method in GENERATOR_METHODS:
                        if h.unit_method == HIFIGAN_NO_GRAD:
                            preds_km = generator_out["unit"]["kmeans_pred"]
                        elif h.unit_method in UNIT_METHODS:
                            preds_km = generator_out["revise_logits"]
                            # preds_km = preds_km[..., :-1]  # rid of padding class idx... WON'T WORK?!
                        preds_km = preds_km.transpose(2, 1)  # (B, C, T)
                    targets_km = avhubert_source_batch[unit_key].to(device)
            text_gf, text_vc = None, None
            if y_g_hat is not None:
                text_gf = audioeval_gf.eval_metrics(y_g_hat, y, wav_padding_mask, gt_texts)
            if y_g_hat_vc is not None:
                text_vc = audioeval_vocoder.eval_metrics(y_g_hat_vc, y, wav_padding_mask, gt_texts, preds_km=preds_km, targets_km=targets_km)
            if tmpdir:
                pbar_desc = f"{saved_samples=}"
            else:
                pbar_desc = f'current wer={audioeval_vocoder.err_tot["wer_vocoder"]}(vc), {audioeval_gf.err_tot["wer"]}(gf)'
            pbar.set_description(pbar_desc)
            if y_g_avhubert_mel is not None:
                err_tot["mel_spec_error_avhubert"] += F.l1_loss(y_mel.masked_select(~mel_padding_mask.unsqueeze(1)), y_g_avhubert_mel.masked_select(~mel_padding_mask.unsqueeze(1))).item()
            if text_gf is not None:
                for line in text_gf:
                    f_gf.write(line+'\n')
            if text_vc is not None:
                for line in text_vc:
                    f_vc.write(line+'\n')
            text = batch["target"]
            for line in text:
                f_gt.write(line)
            if j <= 4:
                # save first few validation sample
                save_wav_16khz(os.path.join(sw.get_logdir(), f'{gt_prefix}_y_{j}.wav'), y[0])
                sw.add_text(f'{gt_prefix}/y_text_{j}', text[0], steps)
                if text_gf is not None:
                    sw.add_text(f'{generated_prefix}/y_text_{j}', text_gf[0], steps)
                if text_vc is not None:
                    sw.add_text(f'{generated_prefix}/y_text_{j}', text_vc[0], steps)
                sw.add_figure(f'{gt_prefix}/y_spec_{j}', plot_spectrogram(y_mel[0].cpu()), steps)
                logging.info(f"saved {j} for {generated_prefix}")
                if y_g_hat is not None:
                    save_wav_16khz(os.path.join(sw.get_logdir(), f'{generated_prefix}_y_hat_griffin_lim{j}.wav'), y_g_hat[0].squeeze())
                if y_g_hat_vc is not None:
                    save_wav_16khz(os.path.join(sw.get_logdir(), f'{generated_prefix}_y_hat_vocoder{j}.wav'), y_g_hat_vc[0].squeeze())
                if y_g_avhubert_mel is not None:
                    sw.add_figure(f'{generated_prefix}/y_hat_vanilla_mel_{j}',
                                    plot_spectrogram(y_g_avhubert_mel[0].squeeze(0).cpu().numpy()), steps)

        del w2v_model, w2v_processor
        err_tot = {**err_tot, **audioeval_gf.err_tot, **audioeval_vocoder.err_tot}
        for err_key, err_term in err_tot.items():
            if err_key == 'algorithmic':
                continue
            if err_key not in err_tot['algorithmic']:
                err_term = err_term / (j+1)
            sw.add_scalar(f"{mode}/{err_key}", err_term, steps)
            metrics[err_key] = err_term
            if mode == VALID_MODE and best_metrics is None:
                best_metrics = metrics
                

def test_eer(
    model,
    device,
    args,
    h,
    sw:SummaryWriter=None,
):
    global steps
    is_main = sw is not None
    eer_metric = EERMetric(device)
    model.eval()
    torch.cuda.empty_cache()
    # TODO magic path is bad
    vox2_avhubert_path = "conf/avhubert/large_avhubert_vox2all.yaml"
    pair_path = "/data1/yfliu/voxceleb2/voxceleb2_testpairs.txt"
    corentinJEncoder.load_model(se_path, device)
    # vox2 test set
    mode = TEST_MODE
    avhubert_config = load_avhubert_config(vox2_avhubert_path)
    testset = load_dataset_eer(
        "test", 
        avhubert_config["task"],
        vid_dict=True,
        pair_path=pair_path,
        permute=False,
        with_image_tsv=True,
        )
    test_loader, _ = get_dataloader(testset, 
                                    batch_size=args.test_batch_size,
                                    num_workers=h.num_gpus, 
                                    dist_sampler=h.num_gpus > 1,
                                    pin_memory=not h.num_gpus > 1,
                                    drop_last=False,
                                    shuffle=False)
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="EER testing in progress...", disable=not is_main)
        for j, batch in enumerate(pbar):
            labels = batch[0]
            batch = batch[1]
            video = batch["net_input"]["source"]["video"].to(device)
            audio = batch["net_input"]["source"]["audio"].to(device)
            mel_padding_mask = batch["net_input"]["padding_mask_mel"].to(device)
            wav_padding_mask = batch["net_input"]["padding_mask_wav"].to(device)
            unit_target = {
                "kmeans_target":None,
                "kmeans_mask":None,
            }
            padding_mask = ~mel_padding_mask
            if h.unit_name is not None:
                kmeans_mask = batch["net_input"]["padding_mask_km"]
                padding_mask = None
                if kmeans_mask is not None:
                    kmeans_mask = kmeans_mask.to(device)
                    unit_target["kmeans_mask"] = ~kmeans_mask
                    padding_mask = ~kmeans_mask
            image_input = batch["net_input"]["source"]["images"]
            if image_input is not None:
                image_input = image_input.to(device)
            waveforms = model(video,
                              farl_img_input=image_input, 
                              vid_masks=padding_mask,
                              audio=audio,  # for svts only
                              )["wav_generated"]  # [B*2, T']
            similarity = corentinJEncoder.compute_similarity(
                waveforms.squeeze(1),
                ~wav_padding_mask,
                max_audio_sample_size=4*16000,  # 4 seconds. Longer is better but consumes more mem.
                pad_audio=False,
                )
            eer_metric.update(
                preds=similarity,
                labels=labels,
            )
    final_eer = eer_metric.compute()
    if is_main:
        print(final_eer)
        sw.add_scalar(f"{mode}/eer", final_eer, steps)


def main():
    
    logging.info('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    default_ckpt_dir = 'cp_hifigan'
    parser.add_argument('--checkpoint_path', default=default_ckpt_dir)
    parser.add_argument('--hifigan_config', default='conf/hifigan/video2speech_template.json')
    parser.add_argument('--avhubert_config', default='conf/avhubert/base_avhubert_30h.yaml')
    parser.add_argument('--avhubert_ckpt', help='if specified, will load pretrained weight onto AVHuBERTModel')
    parser.add_argument('--hifigan_ckpt', help='if specified, will load pretrained weight onto HiFi-GAN in v2w mode'\
        ' as part of the model or in v2m mode (with gradient) as mel-to-audio converter in v2w mode(without gradient)')
    parser.add_argument('--bigvgan_ckpt', help='if specified, will load BigVGAN as mel-to-audio converter in v2w mode.')
    parser.add_argument('--pwg_ckpt', help='if specified, will load Parallel WaveGAN (PWG) as mel-to-audio converter in v2w mode.')
    parser.add_argument('--use_farl', action='store_true', help='if specified, will apply farl in your pretrained unit vocoder')
    parser.add_argument('--skip_ckptcheck', action='store_true', help='if specified, will set strict=False in loading')
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--decay_melloss', action='store_true', help='(deprecated) if specified, will decay mel loss in first 1/5 of total epochs.')
    parser.add_argument('--train_mode', choices=[VIDEO2MEL_MODE, VIDEO2WAV_MODE], default=VIDEO2MEL_MODE, help='v2w(video2wav), v2m(video2mel)')
    parser.add_argument('--test', action='store_true', help='run test only')
    parser.add_argument('--test_all', action='store_true', help='equivalent to --test but with eer test which should be time consuming (Multi-GPU supported).')
    parser.add_argument('--n_ckpts', type=int, default=10, help='number of checkpoints to be saved.')
    parser.add_argument('--save_samples', action='store_true', help='If enabled, will save video-audio synced files in checkpointdir which is used for LSE-C/D and MOS evaluation.')
    parser.add_argument('--svts', action='store_true', help='If specified, will apply SVTS as frontend encoder. avhubert_config is still needed to load dataset.')

    a = parser.parse_args()
    if a.save_samples or a.test_all:
        # Clearly not doing training in this case
        a.test = True
    logging.info(f'Proceeding with train_mode {a.train_mode}')
    if a.checkpoint_path == default_ckpt_dir:
        logging.warning(f"You're using default checkpoint dir {default_ckpt_dir}.\n"+\
            " This should not happen in serious runs as checkpoint dir is likely overwritten with runs in default args.")

    h = load_hifigan_config(a.hifigan_config)
    avhubert_config = load_avhubert_config(a.avhubert_config)
    if '433h_data' in avhubert_config["task"].data or '224h_data' in avhubert_config["task"].data:
        h.total_updates*=8
        h.frozen_steps*=8    
    if a.train_mode == VIDEO2MEL_MODE:
        # give random port to avoid collision
        url = h.dist_config['dist_url']
        splits = url.split(":")
        port = int(splits[-1])
        port -= random.randint(100, 1000)
        h.dist_config['dist_url'] = ':'.join(splits[:-1]+[str(port)])
    # simple hacking for v2m mode
    if a.train_mode == VIDEO2MEL_MODE and h.save_on_metric == 'mel_spec_error_generator':
        h.save_on_metric = "mel_spec_error_avhubert"
    build_env(a.hifigan_config, 'hifigan_config.json', a.checkpoint_path)
    OmegaConf.save(avhubert_config, os.path.join(a.checkpoint_path, 'avhubert_config.yaml'))

    torch.manual_seed(h.seed)
    # TODO magic batch size is bad
    a.test_batch_size = 1
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        # a.batch_size = int(a.batch_size / h.num_gpus)
        a.total_batch_size = h.batch_size * h.num_gpus
        logging.info(f'Batch size per GPU :{h.batch_size}')
        logging.info(f"Total batch size on all GPUs :{a.total_batch_size}")
    else:
        pass
    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h, avhubert_config))
    else:
        train(0, a, h, avhubert_config)


if __name__ == '__main__':
    main()
