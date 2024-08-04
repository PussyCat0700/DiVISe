import argparse
import logging
import math
import os
import random
import time

import torch
from tqdm import tqdm
import wandb
from avhubert.avhubert_as_spkencoder import MoCoSpeakerEncoder, SimCLRSpeakerEncoder
from constants import CONTRAST_METHODS, SIMCLR, MOCO
from contrastive.metrics import EERMetric, calc_cosine_similarity
from dataset.dataset_loading import get_dataloader, load_avhubert_config, load_dataset_contrastive, load_dataset_eer
from torch.distributed import init_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from utils import DataLoaderSeeder, TriStageLRScheduler, load_checkpoint, save_checkpoint, scan_checkpoint, seed_everything
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

metrics = {}
best_metrics = None
steps = 0

def train(rank, a, avhubert_config):
    global metrics, best_metrics, steps
    if rank == 0 and a.wandb:
        full_path = os.path.abspath(a.checkpoint_path)
        pardir = os.path.abspath(f'{full_path}/{os.pardir}')
        proj_name = os.path.basename(pardir)
        run_name = os.path.basename(full_path)
        if a.test:
            run_name += '-test'
        wandb.init(project=proj_name, name=run_name, sync_tensorboard=True)
    
    if a.num_gpus > 1:
        init_process_group(backend="nccl", init_method=a.dist_url,
                           world_size=a.num_gpus, rank=rank)
        
    seed_everything(a.seed)
    torch.cuda.set_device(rank)  # A very strong boost. See https://github.com/jik876/hifi-gan/pull/25
    device = torch.device('cuda:{:d}'.format(rank))
    if a.contrast == SIMCLR:
        speaker_encoder = SimCLRSpeakerEncoder(
            avhubert_config["model"]
            ).to(device)
    elif a.contrast == MOCO:
        speaker_encoder = MoCoSpeakerEncoder(
            avhubert_config["model"],
            K=a.K,
        )
    model_module = speaker_encoder

    if rank == 0:
        logging.info('model loaded.')
        os.makedirs(a.checkpoint_path, exist_ok=True)
        logging.info(f"checkpoints directory : {a.checkpoint_path}")

    os.makedirs(a.checkpoint_path, exist_ok=True)
    cp_g = scan_checkpoint(a.checkpoint_path, 'spkenc_')

    if a.avhubert_ckpt is not None:
        speaker_encoder.load_pretrained(a.avhubert_ckpt, device)
    
    if cp_g is None:
        state_dict = None
        last_epoch = -1
    else:
        state_dict = load_checkpoint(cp_g, device)
        speaker_encoder.load_state_dict(state_dict['model'])
        steps = state_dict['steps'] + 1
        last_epoch = state_dict['epoch']
        best_metrics = state_dict['metrics']
    
    speaker_encoder.to(device)
    if a.num_gpus > 1:
        speaker_encoder = DistributedDataParallel(speaker_encoder, device_ids=[rank], find_unused_parameters=True).to(device)
        model_module = speaker_encoder.module
    optim_g = torch.optim.AdamW([p for p in speaker_encoder.parameters() if p.requires_grad], a.learning_rate, betas=[a.adam_b1, a.adam_b2])
    actual_total_updates = math.ceil(a.total_updates / a.num_gpus)
    part_updates = math.ceil(actual_total_updates / a.n_ckpts)
    saving_updates = {x for x in range(part_updates, actual_total_updates+1, part_updates)}
    logging.info(f"{actual_total_updates=}")
    if not a.test:
        if state_dict is not None:
            optim_g.load_state_dict(state_dict['optim_g'])
        actual_frozen_updates = math.ceil(a.frozen_steps / a.num_gpus)
        if steps <= actual_frozen_updates:
            logging.info(f"AVHuBERT will be frozen for {actual_frozen_updates} updates.")
            model_module.avhubert_grad(False)
        else:
            logging.info(f"current {steps=}. AVHuBERT will not be frozen after {actual_frozen_updates} updates.")
        # Exactly as in ReVISE Tab. 17
        scheduler_g = TriStageLRScheduler(optim_g, actual_total_updates, a.t1_percent, a.t2_percent, last_lr_factor=a.last_lr_factor,last_epoch=steps-1)
        model_module.update_steps(steps, actual_total_updates)
    
    trainset = load_dataset_contrastive("train", avhubert_config["task"])
    train_loader, train_sampler = get_dataloader(trainset, 
                                                batch_size=a.batch_size,
                                                num_workers=a.num_gpus, 
                                                dist_sampler=a.num_gpus > 1,
                                                pin_memory=not a.num_gpus > 1,
                                                shuffle=True,
                                                seeder=DataLoaderSeeder(1234),
                                                )
    sw = None
    if rank == 0:
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))
    # vox2 valid set
    # TODO maybe shift everything to our new dataset? 
    validset = load_dataset_contrastive("valid", avhubert_config["task"])
    validation_loader, _ = get_dataloader(validset, 
                                        batch_size=a.batch_size,
                                        num_workers=a.num_gpus, 
                                        dist_sampler=a.num_gpus > 1,
                                        pin_memory=not a.num_gpus > 1,
                                        drop_last=False,
                                        shuffle=False)
    # vox2 test set
    testset = load_dataset_eer(
        "test", 
        avhubert_config["task"],
        vid_dict=True,
        pair_path="/data1/yfliu/voxceleb2/voxceleb2_testpairs.txt")
    test_loader, _ = get_dataloader(testset, 
                                    batch_size=a.test_batch_size,
                                    num_workers=a.num_gpus, 
                                    dist_sampler=a.num_gpus > 1,
                                    pin_memory=not a.num_gpus > 1,
                                    drop_last=False,
                                    shuffle=False)
    # val args
    val_args = {
        "speaker_encoder":speaker_encoder,
        "device":device,
        "sw":sw,
    }

    a.training_epochs = math.ceil(a.total_updates / a.num_gpus / len(train_loader))
    logging.info(f"{a.training_epochs=}")
    if not a.test:
        for epoch in range(max(0, last_epoch), a.training_epochs):
            if rank == 0:
                start = time.time()
                logging.info("Epoch: {}".format(epoch+1))

            if a.num_gpus > 1:
                train_sampler.set_epoch(epoch)
            pbar = tqdm(train_loader)
            for batch in pbar:
                speaker_encoder.train()
                if rank == 0:
                    start_b = time.time()
                avhubert_source_batch = batch["net_input"]["source"]
                loss = speaker_encoder(avhubert_source_batch["video"].to(device))

                optim_g.zero_grad()
                loss.backward()
                optim_g.step()

                if rank == 0:
                    # STDOUT logging
                    if steps % a.stdout_interval == 0:
                        pbar.set_description('Epoch: {:d}, CE Loss Total : {:4.3f}, s/b : {:4.3f}'.
                            format(epoch, loss, time.time() - start_b))

                    # Tensorboard summary logging
                    if steps % a.summary_interval == 0:
                        def log_training(tag, value):
                            sw.add_scalar(f"training/{tag}", value, steps)
                        log_training("ce_loss", loss)
                        log_training("epoch", epoch)

                steps += 1
                # scheduler is updated step-level
                scheduler_g.step()
                if rank == 0:
                    if steps == actual_frozen_updates:
                        model_module.avhubert_grad(True)
                        sw.add_scalar(f"training/unfreeze_step", steps, steps)
                    # Validation&Checkpointing
                    if steps in saving_updates:
                        def save_all_checkpoints(save_title, remove_title=None):
                            checkpoint_path = "{}/spkenc_{}".format(a.checkpoint_path, save_title)
                            prev_checkpoint_path_g = "{}/spkenc_{}".format(a.checkpoint_path, remove_title) if remove_title is not None else None
                            save_checkpoint(checkpoint_path,
                                            {'model': (speaker_encoder.module if a.num_gpus > 1 else speaker_encoder).state_dict(),
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
                if steps in saving_updates:
                    val_args["loader"] = validation_loader
                    validate(**val_args)
                if steps == actual_total_updates:
                    # Early breaking
                    break
            
            if rank == 0:
                logging.info('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))
            # End of a train epoch
    else:
        val_args["loader"] = test_loader
        test_eer(**val_args)
        
    if rank == 0 and a.wandb:
        wandb.finish()
 

def validate(
    speaker_encoder,
    device,
    loader,
    sw:SummaryWriter=None,
    ):
    global steps
    is_main = sw is not None
    speaker_encoder.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation in progress...", disable=not is_main)
        tot_loss = 0
        for j, batch in enumerate(pbar):
            avhubert_source_batch = batch["net_input"]["source"]
            loss = speaker_encoder(avhubert_source_batch["video"].to(device))
             # Reduce the loss from all GPUs to get the sum/average properly
            reduced_loss = loss.clone()
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            tot_loss += reduced_loss.item()
        if is_main:
            avg_loss = tot_loss / len(loader.dataset)  # Make sure to divide by total dataset size
            sw.add_scalar(f"valid/ce_loss", avg_loss, steps)
            

def test_eer(
    speaker_encoder,
    device,
    loader,
    sw:SummaryWriter=None,
):
    global steps
    is_main = sw is not None
    eer_metric = EERMetric(device)
    speaker_encoder.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        pbar = tqdm(loader, desc="EER testing in progress...", disable=not is_main)
        for j, batch in enumerate(pbar):
            labels = batch[0]
            batch = batch[1]
            out = speaker_encoder(
                batch["net_input"]["source"]["video"].to(device),
                detailed=True,
                )["embeddings"]
            similarity = calc_cosine_similarity(out)
            eer_metric.update(
                preds=similarity,
                labels=labels,
            )
    final_eer = eer_metric.compute()
    if is_main:
        sw.add_scalar(f"testing/eer", final_eer, steps)
        print(final_eer)
    
    
def main():
    
    logging.info('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    default_ckpt_dir = 'cp_contrastive'
    parser.add_argument('--checkpoint_path', default=default_ckpt_dir)
    parser.add_argument('--avhubert_config', default='conf/avhubert/large_avhubert_vox2all.yaml')
    parser.add_argument('--avhubert_ckpt', help='if specified, will load pretrained weight onto AVHuBERTModel')
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--test', action='store_true', help='run test only')
    parser.add_argument('--n_ckpts', type=int, default=10, help='number of checkpoints to be saved.')
    parser.add_argument('--contrast', choices=CONTRAST_METHODS, required=True, help='Contrastive Learning Strategies')

    a = parser.parse_args()
    
    # TODO temp constant vars
    a.seed = 42
    a.frozen_steps = -100  # no freezing currently
    # a.total_updates = 600000  # a total of 600K updates
    a.total_updates = 45000 * 8
    a.learning_rate = 6e-5
    a.adam_b1 = 0.9
    a.adam_b2 = 0.98
    a.t1_percent = 10
    a.t2_percent = 20
    a.last_lr_factor = 0.05
    a.batch_size = 10
    a.test_batch_size = 1

    if a.checkpoint_path == default_ckpt_dir:
        logging.warning(f"You're using default checkpoint dir {default_ckpt_dir}.\n"+\
            " This should not happen in serious runs as checkpoint dir is likely overwritten with runs in default args.")
    
    avhubert_config = load_avhubert_config(a.avhubert_config)
    if a.contrast == MOCO:
        # Should be mod 0 of total batch size
        a.K = 320
        
    logging.info(a)
    a.dist_url = 'tcp://localhost'
    port = 55000+random.randint(100, 1000)
    a.dist_url = a.dist_url + f':{port}'
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(a.seed)
        a.num_gpus = torch.cuda.device_count()
        a.total_batch_size = a.batch_size * a.num_gpus
        logging.info(f'Batch size per GPU :{a.batch_size}')
        logging.info(f"Total batch size on all GPUs :{a.total_batch_size}")
    else:
        pass
    if a.num_gpus > 1:
        mp.spawn(train, nprocs=a.num_gpus, args=(a, avhubert_config))
    else:
        train(0, a, avhubert_config)

if __name__ == '__main__':
    main()