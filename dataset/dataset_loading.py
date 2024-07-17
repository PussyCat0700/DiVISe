from argparse import Namespace

from omegaconf import OmegaConf
from avhubert import AVHubertPretrainingConfig, AVHubertConfig
from constants import UNIT_SPEECH_TOKENIZER_NO_GRAD
from dataset.avdatasets import AVHubertDataset, ContrastiveDataset
from torch.utils.data import DistributedSampler, DataLoader, ConcatDataset
from typing import List
import logging

from dataset.eer import eer_dataset

logger = logging.getLogger(__name__)

def load_avhubert_config(cfg_path):
    file_config = OmegaConf.load(cfg_path)
    task_config = OmegaConf.merge(AVHubertPretrainingConfig(), file_config["task"])
    model_config = OmegaConf.merge(AVHubertConfig(), file_config["model"])
    return {
        "task": task_config,
        "model": model_config,
        }

class AVHuBERTAdaptingCollater:
    """Collater designed especially for fairseq's dataset
    so that it can fit into torch Dataloader
    """
    def __init__(self, dataset:AVHubertDataset):
        self.dataset = dataset
    
    def __call__(self, samples):
        if isinstance(self.dataset, ConcatDataset):
            batch = self.dataset.datasets[0].collater(samples=samples)
        else:
            batch = self.dataset.collater(samples=samples)
        return batch

def get_dataloader(dataset, batch_size, shuffle, num_workers=0, dist_sampler=False, pin_memory=True, seeder=None, drop_last=True):
    if dist_sampler:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = None  # sampler option is mutually exclusive with shuffle
    else:
        sampler = None
    collate_fn_adapter = None    
    if isinstance(dataset, eer_dataset.ContrastivePairTestDataset):
        collate_fn_adapter = dataset.collater
    else:
        collate_fn_adapter = AVHuBERTAdaptingCollater(dataset)
    loader = DataLoader(dataset, num_workers=num_workers, shuffle=shuffle,
                              sampler=sampler,
                              batch_size=batch_size,
                              pin_memory=pin_memory,
                              drop_last=drop_last,
                              collate_fn=collate_fn_adapter,
                              worker_init_fn=seeder)
    
    return loader, sampler

def load_dataset(split: str, 
                 cfg:AVHubertPretrainingConfig, 
                 pitch_type=None,
                 km_name=None,
                 hu_name=None,
                 st_type=None,
                 generator_mode=None,
                 fake_km_mask=False,
                 km_pad_class_idx=None,
                 max_keep_sample_size=500,
                 vid_dict=False,
                 with_image_tsv=False,
                 with_text=True,
                 ) -> None:
        manifest = f"{cfg.data}/{split}.tsv"
        paths = None
        image_tsv_path = None
        if with_image_tsv:
            image_tsv_path = f"{cfg.data}/frame_{split}.tsv"
        if with_text:
            paths = [
                f"{cfg.data}/{split}.{l}" for l in cfg.labels
            ]
        if km_name is not None:
            if generator_mode == UNIT_SPEECH_TOKENIZER_NO_GRAD:
                km_name = None
            else:
                km_name = f"{cfg.data}/{km_name}.km"
        image_aug = cfg.image_aug if split == 'train' else False
        # noise_fn, noise_snr = f"{self.cfg.noise_wav}/{split}.tsv" if self.cfg.noise_wav is not None else None, eval(self.cfg.noise_snr)
        # noise_num = self.cfg.noise_num
        max_sample_seconds=cfg.max_sample_seconds
        pad_audio=cfg.pad_audio
        random_crop=cfg.random_crop
        modalities=cfg.modalities
        dataset = AVHubertDataset(
            manifest,  #  a path list where you store your tsv files
            sample_rate=cfg.sample_rate,  # 16000 (constant)
            label_paths=paths,  # a path list where you store your dictionaries
            max_keep_sample_size=max_keep_sample_size,
            min_keep_sample_size=cfg.min_sample_size,  # None
            max_sample_seconds=max_sample_seconds,
            pitch_type=pitch_type,
            km_path=km_name,
            st_type=st_type,
            km_pad_class=km_pad_class_idx,
            hu_name=hu_name,
            fake_km_mask=fake_km_mask,
            pad_audio=pad_audio,  # Should be True
            normalize=cfg.normalize,  # True
            store_labels=False,
            random_crop=random_crop,  #  Should be False in decoding
            single_target=cfg.single_target,  # True
            image_mean=cfg.image_mean,  # 0.421
            image_std=cfg.image_std,  # 0.165
            image_crop_size=cfg.image_crop_size,  # 88
            image_aug=image_aug,  # False in infernece
            modalities=modalities,  # if your modality setting doesn't work, this might be where to find a clue.
            vid_dict=vid_dict,  # set to True when you want to test video EER on voxceleb2
            image_tsv_path=image_tsv_path,  # when you want to use farl, this should be set
            # noise_fn=noise_fn,
            # noise_prob=cfg.noise_prob,  # 0.0
            # noise_snr=noise_snr,
            # noise_num=noise_num
        )
        return dataset


def load_dataset_eer(split: str, 
                 cfg:AVHubertPretrainingConfig, 
                 vid_dict: bool,
                 pair_path: str,
                 ) -> eer_dataset.VideoPairDataset:
        manifest = f"{cfg.data}/{split}.tsv"
        image_aug = cfg.image_aug if split == 'train' else False
        # noise_fn, noise_snr = f"{self.cfg.noise_wav}/{split}.tsv" if self.cfg.noise_wav is not None else None, eval(self.cfg.noise_snr)
        # noise_num = self.cfg.noise_num
        pad_audio=cfg.pad_audio
        random_crop=cfg.random_crop
        modalities=cfg.modalities
        dataset = eer_dataset.VideoPairDataset(
            manifest,  #  a path list where you store your tsv files
            sample_rate=cfg.sample_rate,  # 16000 (constant)
            label_paths=None,  # a path list where you store your dictionaries
            max_keep_sample_size=100000,
            min_keep_sample_size=cfg.min_sample_size,  # None
            max_sample_seconds=100000,
            pad_audio=pad_audio,  # Should be True
            normalize=cfg.normalize,  # True
            store_labels=False,
            random_crop=random_crop,  #  Should be False in decoding
            single_target=cfg.single_target,  # True
            image_mean=cfg.image_mean,  # 0.421
            image_std=cfg.image_std,  # 0.165
            image_crop_size=cfg.image_crop_size,  # 88
            image_aug=image_aug,  # False in infernece
            modalities=modalities,  # if your modality setting doesn't work, this might be where to find a clue.
            vid_dict=vid_dict,  # set to True when you want to test video EER on voxceleb2
            pair_path=pair_path,
            # noise_fn=noise_fn,
            # noise_prob=cfg.noise_prob,  # 0.0
            # noise_snr=noise_snr,
            # noise_num=noise_num
        )
        return dataset

    
def load_dataset_contrastive(split: str, cfg:AVHubertPretrainingConfig, max_keep_sample_size=500) -> None:
    manifest = f"{cfg.data}/{split}.tsv"
    image_aug = cfg.image_aug if split == 'train' else False
    # noise_fn, noise_snr = f"{self.cfg.noise_wav}/{split}.tsv" if self.cfg.noise_wav is not None else None, eval(self.cfg.noise_snr)
    # noise_num = self.cfg.noise_num
    max_sample_seconds=cfg.max_sample_seconds
    pad_audio=cfg.pad_audio
    random_crop=cfg.random_crop
    modalities=cfg.modalities
    dataset = ContrastiveDataset(
        manifest,  #  a path list where you store your tsv files
        sample_rate=cfg.sample_rate,  # 16000 (constant)
        max_keep_sample_size=max_keep_sample_size,
        min_keep_sample_size=cfg.min_sample_size,  # None
        max_sample_seconds=max_sample_seconds,
        pad_audio=pad_audio,  # Should be True
        normalize=cfg.normalize,  # True
        random_crop=random_crop,  #  Should be False in decoding
        single_target=cfg.single_target,  # True
        image_mean=cfg.image_mean,  # 0.421
        image_std=cfg.image_std,  # 0.165
        image_crop_size=cfg.image_crop_size,  # 88
        image_aug=image_aug,  # False in infernece
        modalities=modalities,  # if your modality setting doesn't work, this might be where to find a clue.
        # noise_fn=noise_fn,
        # noise_prob=cfg.noise_prob,  # 0.0
        # noise_snr=noise_snr,
        # noise_num=noise_num
    )
    return dataset