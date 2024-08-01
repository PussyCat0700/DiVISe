import argparse
import logging
import math
import os
from pathlib import Path
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import wandb
from constants import HIFIGAN_NO_GRAD

from contrastive.metrics import EERMetric
import speaker_encoder.inference as corentinJEncoder
from dataset.dataset_loading import get_dataloader, load_avhubert_config, load_dataset_eer, load_hifigan_config, load_dataset
from dataset.meldataset import LogMelSpectrogram
from models import Generator as HifiganGenerator
from custom_hifigan.hifigan.discriminator import (
    HifiganDiscriminator,
    feature_loss,
    discriminator_loss,
    generator_loss,
)
from custom_hifigan.hifigan.utils import load_checkpoint, save_checkpoint, plot_spectrogram
from utils import DataLoaderSeeder


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


SEGMENT_LENGTH = 8320
HOP_LENGTH = 160
HOP_LENGTH_HUBERT = 320
KMEANS_CLASSES = 2000
SAMPLE_RATE = 16000
BASE_LEARNING_RATE = 2e-4
FINETUNE_LEARNING_RATE = 1e-4
BETAS = (0.8, 0.99)
LEARNING_RATE_DECAY = 0.999
WEIGHT_DECAY = 1e-5
LOG_INTERVAL = 5
VALIDATION_INTERVAL = 1000
NUM_GENERATED_EXAMPLES = 10
CHECKPOINT_INTERVAL = 5000


def train_model(rank, world_size, args, avhubert_config, hifigan_config):
    device = torch.device(f'cuda:{rank}')
    use_farl = args.farl_ckpt is not None
    logmel = LogMelSpectrogram().to(rank)
    if hifigan_config.num_gpus > 1:
        dist.init_process_group(
            backend=hifigan_config.dist_config['dist_backend'],
            init_method=hifigan_config.dist_config['dist_url'],
            world_size=hifigan_config.num_gpus,
            rank=rank
        )
    if rank==0 and args.wandb:
        full_path = os.path.abspath(args.checkpoint_dir)
        pardir = os.path.abspath(f'{full_path}/{os.pardir}')
        proj_name = os.path.basename(pardir)
        run_name = os.path.basename(full_path)
        wandb.init(project=proj_name, name=run_name, sync_tensorboard=True)

    log_dir = args.checkpoint_dir / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    if rank == 0:
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(log_dir / f"{args.checkpoint_dir.stem}.log")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.setLevel(logging.ERROR)

    writer = SummaryWriter(log_dir) if rank == 0 else None

    generator = HifiganGenerator(hifigan_config, hifigan_config.num_mels,
                                    unit_nums=hifigan_config.k,
                                    use_farl=use_farl,
                                    ).to(rank)
    discriminator = HifiganDiscriminator().to(rank)
    optimizer_generator = optim.AdamW(
        generator.parameters(),
        lr=BASE_LEARNING_RATE if not args.finetune else FINETUNE_LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer_discriminator = optim.AdamW(
        discriminator.parameters(),
        lr=BASE_LEARNING_RATE if not args.finetune else FINETUNE_LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler_generator = optim.lr_scheduler.ExponentialLR(
        optimizer_generator, gamma=LEARNING_RATE_DECAY
    )
    scheduler_discriminator = optim.lr_scheduler.ExponentialLR(
        optimizer_discriminator, gamma=LEARNING_RATE_DECAY
    )
    if use_farl:
        generator.load_pretrained_farlmodel(args.farl_ckpt, map_location=device)
    if args.resume is not None:
        global_step, best_loss = load_checkpoint(
            load_path=args.resume,
            generator=generator,
            discriminator=discriminator,
            optimizer_generator=optimizer_generator,
            optimizer_discriminator=optimizer_discriminator,
            scheduler_generator=scheduler_generator,
            scheduler_discriminator=scheduler_discriminator,
            rank=rank,
            logger=logger,
            finetune=args.finetune,
        )
    else:
        global_step, best_loss = 0, float("inf")

    if args.finetune:
        global_step, best_loss = 0, float("inf")

    if hifigan_config.num_gpus > 1:
        generator = DDP(generator, device_ids=[rank])
        discriminator = DDP(discriminator, device_ids=[rank])

    dataloading_kwargs = {}
    if hifigan_config.unit_name is not None:
        dataloading_kwargs = {
            "pitch_type":hifigan_config.prosody_type,
            "km_name":hifigan_config.unit_name,
            "hu_name":hifigan_config.hu_repr_name,
            "st_type":hifigan_config.st_type,
            "km_pad_class_idx": hifigan_config.k,
            "generator_mode":HIFIGAN_NO_GRAD,
            "with_image_tsv": use_farl,
            "with_text": False,
        }
    original_max_sample_seconds = avhubert_config["task"].max_sample_seconds
    trainset = load_dataset("train", avhubert_config["task"], **dataloading_kwargs)
    avhubert_config["task"].max_sample_seconds = original_max_sample_seconds
    train_loader, train_sampler = get_dataloader(trainset, 
                                                batch_size=hifigan_config.batch_size,
                                                num_workers=hifigan_config.num_gpus, 
                                                dist_sampler=hifigan_config.num_gpus > 1,
                                                pin_memory=not hifigan_config.num_gpus > 1,
                                                shuffle=True,
                                                seeder=DataLoaderSeeder(hifigan_config.seed),
                                                )
    if rank == 0:
        kwargs = {
            "st_type": hifigan_config.st_type
        }
        if hifigan_config.unit_name is not None and hifigan_config.valid_unit_name is not None:
            # You can apply trained kmeans model on valid set to get km labels just for reference.
            kwargs.update({
                "km_name":hifigan_config.valid_unit_name,
                })
        else:
            kwargs.update({
                "fake_km_mask":True,
            })
        dataloading_kwargs.update(**kwargs)
        validset = load_dataset("valid", avhubert_config["task"], **dataloading_kwargs)
        validation_loader, _ = get_dataloader(validset, 
                                            batch_size=hifigan_config.batch_size,
                                            num_workers=1, 
                                            drop_last=False,
                                            shuffle=False)
        if hifigan_config.unit_name is not None and hifigan_config.test_unit_name is not None:
            # You can apply trained kmeans model on valid set to get km labels just for reference.
            kwargs.update({
                "km_name":hifigan_config.test_unit_name,
                })
        else:
            kwargs.update({
                "fake_km_mask":True,
            })

    melspectrogram = LogMelSpectrogram().to(rank)

    if args.test:
        logger.info("will only perform test")
        test_eer(
            model=generator,
            device=device,
            h=hifigan_config,
            global_steps=global_step,
            sw=writer
        )
        exit(0)
    n_epochs = math.ceil(args.max_updates/(len(train_loader)*world_size))
    start_epoch = global_step // len(train_loader) + 1

    logger.info("**" * 40)
    logger.info(f"batch size: {hifigan_config.batch_size}")
    logger.info(f"iterations per epoch: {len(train_loader)}")
    logger.info(f"total of epochs: {n_epochs}")
    logger.info(f"started at epoch: {start_epoch}")
    logger.info("**" * 40 + "\n")

    for epoch in range(start_epoch, n_epochs + 1):
        if hifigan_config.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        generator.train()
        discriminator.train()
        average_loss_mel = average_loss_discriminator = average_loss_generator = 0
        pbar = tqdm(train_loader)
        for i, batch in enumerate(pbar, 1):
            avhubert_source_batch = batch["net_input"]["source"]
            wavs = avhubert_source_batch["audio"].to(rank)
            wavs = wavs.unsqueeze(1)
            tgts = logmel(wavs)
            units = avhubert_source_batch["km"].to(rank)
            image_inputs = None
            if use_farl:
                image_inputs = avhubert_source_batch["images"].to(rank)

            # Discriminator
            optimizer_discriminator.zero_grad()

            wavs_ = generator(units, image_inputs)
            mels_ = melspectrogram(wavs_)
            scores, _ = discriminator(wavs)
            scores_, _ = discriminator(wavs_.detach())

            loss_discriminator, _, _ = discriminator_loss(scores, scores_)

            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Generator
            optimizer_generator.zero_grad()

            scores, features = discriminator(wavs)
            scores_, features_ = discriminator(wavs_)

            loss_mel = F.l1_loss(mels_, tgts)
            loss_features = feature_loss(features, features_)
            loss_generator_adversarial, _ = generator_loss(scores_)
            loss_generator = 45 * loss_mel + loss_features + loss_generator_adversarial

            loss_generator.backward()
            optimizer_generator.step()

            global_step += 1

            average_loss_mel += (loss_mel.item() - average_loss_mel) / i
            average_loss_discriminator += (
                loss_discriminator.item() - average_loss_discriminator
            ) / i
            average_loss_generator += (
                loss_generator.item() - average_loss_generator
            ) / i

            if rank == 0:
                if global_step % LOG_INTERVAL == 0:
                    writer.add_scalar(
                        "train/loss_mel",
                        loss_mel.item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/loss_generator",
                        loss_generator.item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/loss_discriminator",
                        loss_discriminator.item(),
                        global_step,
                    )

            if global_step % VALIDATION_INTERVAL == 0 and rank == 0:
                generator.eval()

                average_validation_loss = 0
                for j, batch in enumerate(validation_loader, 1):
                    avhubert_source_batch = batch["net_input"]["source"]
                    wavs = avhubert_source_batch["audio"].to(rank)
                    wavs = wavs.unsqueeze(1)
                    tgts = logmel(wavs)
                    units = avhubert_source_batch["km"].to(rank)
                    image_inputs = None
                    if use_farl:
                        image_inputs = avhubert_source_batch["images"].to(rank)
                    with torch.no_grad():
                        wavs_ = generator(units, image_inputs)
                        mels_ = melspectrogram(wavs_)

                        length = min(mels_.size(-1), tgts.size(-1))

                        loss_mel = F.l1_loss(mels_[..., :length], tgts[..., :length])

                    average_validation_loss += (
                        loss_mel.item() - average_validation_loss
                    ) / j

                    if rank == 0:
                        if j <= NUM_GENERATED_EXAMPLES:
                            writer.add_audio(
                                f"generated/wav_{j}",
                                wavs_.squeeze()[0],
                                global_step,
                                sample_rate=16000,
                            )
                            writer.add_figure(
                                f"generated/mel_{j}",
                                plot_spectrogram(mels_.squeeze()[0].cpu().numpy()),
                                global_step,
                            )

                generator.train()
                discriminator.train()

                if rank == 0:
                    writer.add_scalar(
                        "validation/mel_loss", average_validation_loss, global_step
                    )
                    logger.info(
                        f"valid -- epoch: {epoch}, mel loss: {average_validation_loss:.4f}"
                    )

                new_best = best_loss > average_validation_loss
                if new_best or global_step % CHECKPOINT_INTERVAL == 0:
                    if new_best:
                        logger.info("-------- new best model found!")
                        best_loss = average_validation_loss

                    if rank == 0:
                        save_checkpoint(
                            checkpoint_dir=args.checkpoint_dir,
                            generator=generator,
                            discriminator=discriminator,
                            optimizer_generator=optimizer_generator,
                            optimizer_discriminator=optimizer_discriminator,
                            scheduler_generator=scheduler_generator,
                            scheduler_discriminator=scheduler_discriminator,
                            step=global_step,
                            loss=average_validation_loss,
                            best=new_best,
                            logger=logger,
                        )

        scheduler_discriminator.step()
        scheduler_generator.step()

        logger.info(
            f"train -- epoch: {epoch}, mel loss: {average_loss_mel:.4f}, generator loss: {average_loss_generator:.4f}, discriminator loss: {average_loss_discriminator:.4f}"
        )

    dist.destroy_process_group()



def test_eer(
    model,
    device,
    h,
    global_steps,
    sw:SummaryWriter=None,
):
    is_main = sw is not None
    eer_metric = EERMetric()
    model.eval()
    torch.cuda.empty_cache()
    # TODO magic path is bad
    from pathlib import Path
    se_path = Path("/data1/yfliu/model/CorentinJ/encoder.pt")
    vox2_avhubert_path = "conf/avhubert/large_avhubert_vox2all.yaml"
    pair_path = "/data1/yfliu/voxceleb2/voxceleb2_testpairs.txt"
    corentinJEncoder.load_model(se_path, device)
    # vox2 test set
    avhubert_config = load_avhubert_config(vox2_avhubert_path)
    testset = load_dataset_eer(
        "test", 
        avhubert_config["task"],
        vid_dict=True,
        pair_path=pair_path,
        permute=False,
        with_image_tsv=True,
        km_name=h.test_unit_name,
        km_pad_class_idx=h.k,
        )
    test_loader, _ = get_dataloader(testset, 
                                    batch_size=4,
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
            wav_padding_mask = batch["net_input"]["padding_mask_wav"].to(device)
            avhubert_source_batch = batch["net_input"]["source"]
            image_input = avhubert_source_batch["images"].to(device)
            units = avhubert_source_batch["km"].to(device)
            if image_input is not None:
                image_input = image_input.to(device)
            waveforms = model(units, image_input)  # [B*2, T']
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
        sw.add_scalar(f"test/eer", final_eer, global_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or finetune HiFi-GAN.")
    parser.add_argument(
        "--checkpoint_dir",
        default='/data1/yfliu/outputs/unithifigan_test',
        help="path to the checkpoint directory",
        type=Path,
    )
    parser.add_argument(
        "--resume",
        help="path to the checkpoint to resume from",
        type=Path,
    )
    parser.add_argument(
        "--finetune",
        help="whether to finetune (note that a resume path must be given)",
        action="store_true",
    )
    parser.add_argument(
        "--wandb",
        help="wandb enable sign",
        action='store_true',
    )
    parser.add_argument(
        '--hifigan_config',
        default='conf/hifigan/video2speech_revise_original.json'
    )  # TODO: Change back in formal release
    parser.add_argument(
        '--avhubert_config',
        default='conf/avhubert/large_avhubert.yaml'
    )
    parser.add_argument(
        '--farl_ckpt', 
        help='if specified, will apply farl to boost speaker identity information'
    )
    parser.add_argument(
        '--test',
        help='if specified, will only test eer',
        action='store_true',
    )
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # display training setup info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
    logger.info(f"CUDNN enabled: {torch.backends.cudnn.enabled}")
    logger.info(f"CUDNN deterministic: {torch.backends.cudnn.deterministic}")
    logger.info(f"CUDNN benchmark: {torch.backends.cudnn.benchmark}")
    logger.info(f"# of GPUS: {torch.cuda.device_count()}")

    # clear handlers
    logger.handlers.clear()

    world_size = torch.cuda.device_count()
    max_updates_allowed = 400_000
    avhubert_config = load_avhubert_config(args.avhubert_config)
    avhubert_config["task"].max_sample_seconds = SEGMENT_LENGTH / SAMPLE_RATE
    hifigan_config = load_hifigan_config(args.hifigan_config)
    hifigan_config.num_gpus = world_size
    args.max_updates = max_updates_allowed
    if world_size > 1:
        # give random port to avoid collision
        url = hifigan_config.dist_config['dist_url']
        splits = url.split(":")
        port = int(splits[-1])
        port -= random.randint(100, 1000)
        hifigan_config.dist_config['dist_url'] = ':'.join(splits[:-1]+[str(port)])
        mp.spawn(
            train_model,
            args=(world_size, args, avhubert_config, hifigan_config),
            nprocs=world_size,
            join=True,
        )
    else:
        train_model(0, world_size, args, avhubert_config, hifigan_config)