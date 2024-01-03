from omegaconf import OmegaConf
from avhubert.avhubert import AVHubertConfig, AVHubertModel
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq.dataclass.configs import FairseqDataclass
from typing import Dict, List, Optional, Tuple
from omegaconf import MISSING, II
from prosody_predictor.predictor import HuBERTPredictor, HuBERTRepresentationPredictor, HuBERTSoftContentPredictor, ProsodyPredictor

from pytorch_backend.transformer.encoder import ConformerEncoder

@dataclass
class AVHubertPretrainingConfig(FairseqDataclass):
    data: str = field(
        default=MISSING, metadata={"help": "path to data directory"}
    )
    labels: List[str] = field(
        default_factory=lambda: ["ltr"],
        metadata={
            "help": (
                "extension of the label files to load, frame-level labels for"
                " pre-training, and sequence-level label for fine-tuning"
            )
        },
    )
    label_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, looks for labels in this directory instead",
        },
    )
    label_rate: int = field(
        default=-1,
        metadata={"help": "label frame rate. -1 for sequence label"},
    )

    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={
            "help": "if set, normalizes input to have 0 mean and unit variance"
        },
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_sample_seconds: Optional[float] = field(
        default=None,
        metadata={"help": "max sample seconds to keep in training"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to keep in training"},
    )
    max_trim_sample_seconds: Optional[int] = field(
        default=II("task.max_sample_seconds"),
        metadata={"help": "max sample seconds to trim to for batching"},
    )
    single_target: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, AddTargetDatasets outputs same keys "
            "as AddTargetDataset"
        },
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )
    pdb: Optional[bool] = field(
        default=False,
        metadata={"help": "pdb"},
    )
    stack_order_audio: int = field(
        default=1,
        metadata={"help": "concatenate n consecutive audio frames for one step"},
    )
    skip_verify: Optional[bool] = field(
        default=False,
        metadata={"help": "skip verifying label-audio alignment"},
    )
    image_aug: bool = field(default=False, metadata={'help': 'image data augmentation'})
    image_crop_size: int = field(
        default=88, metadata={"help": "image ROI size"})
    image_mean: float = field(
        default=0.421, metadata={"help": "image mean"})
    image_std: float = field(
        default=0.165, metadata={"help": "image std"})
    modalities: Optional[List[str]] = field(default_factory=lambda: ["audio", "video"], metadata={'help': 'modalities to load'})
    is_s2s: bool=field(default=False, metadata={'help': 'seq2seq fine-tuning only'})
    tokenizer_bpe_name: Optional[str] = field(default=None, metadata={'help': 'tokenizer model name'})
    tokenizer_bpe_model: Optional[str] = field(default=None, metadata={'help': 'tokenizer model path'})
    noise_wav: Optional[str] = field(default=None, metadata={'help': 'manifest of noise wav files (one wav file path per line)'})
    noise_prob: float = field(default=0, metadata={'help': 'noise probability'})
    noise_snr: Optional[str] = field(default='0', metadata={'help': 'noise SNR in audio'})
    noise_num: int = field(default=1, metadata={'help': 'number of noise wav files to mix'})
    fine_tuning: bool = field(default=False, metadata={"help": "set to true if fine-tuning AV-Hubert"})

class AVHubertEncoder(nn.Module):
    # should be consistent with conformer's setting.
    lookup_table = {
        "S":{
            "attention_dim":144,
        },
        "M":{
            "attention_dim":256,
        },
        "L":{
            "attention_dim":512,
        }
    }
    def __init__(self, cfg, num_mels, prosody_minmax_dict, unit_dict, hu_dict, size="M", mel_before_conformer=False) -> None:
        super().__init__()
        self.mel_before_conformer = mel_before_conformer
        self.attention_dim = self.lookup_table[size]["attention_dim"]
        self.avhubert_model = AVHubertModel(cfg=cfg)
        self.use_prosody = prosody_minmax_dict is not None
        self.use_hubert_units = unit_dict is not None
        self.use_hubert_representation = hu_dict is not None
        if self.use_prosody:
            self.prosody_predictor = ProsodyPredictor(encoder_hidden=self.attention_dim, **prosody_minmax_dict)
        if self.use_hubert_units:
            self.is_soft = unit_dict.pop("is_soft")
            if self.is_soft:
                self.unit_predictor = HuBERTSoftContentPredictor(**unit_dict)
            else:
                self.unit_predictor = HuBERTPredictor(**unit_dict)
        if self.use_hubert_representation:
            self.hu_predictor = HuBERTRepresentationPredictor(**hu_dict)
        self.conformer_encoder = ConformerEncoder(size)
        self.avhubert2downstream = torch.nn.Linear(768, self.attention_dim*4)
        self.attention2mel = torch.nn.Linear(self.attention_dim, num_mels)
        
    def update_steps(self, current_step, total_steps):
        if self.use_hubert_representation:
            self.hu_predictor.reset_prob(current_step, total_steps)
    
    def forward(self, source, prosody_target, unit_target, hu_target, mel_mask=None):
        # source should only include video
        encoder_out, feature, mask = self.avhubert_model.extract_finetune_with_feature(source)  # (bs, vidlen, 768)
        encoder_out = self.avhubert2downstream(encoder_out)  # (bs, vidlen, attention_dim*4)
        if self.use_hubert_units or self.use_hubert_representation:
            encoder_out = encoder_out.reshape(*encoder_out.shape[:-2], -1, self.attention_dim*2)
            if self.use_hubert_units:
                unit_info = self.unit_predictor(encoder_out, **unit_target)
                encoder_out = unit_info['output']
            if self.use_hubert_representation:
                hu_info = self.hu_predictor(encoder_out, **hu_target)
                encoder_out = hu_info['output']
        # (bs, vidlen, attention_dim*4) -> (bs, mellen=4*vidlen, attention_dim)
        encoder_out = encoder_out.reshape(*encoder_out.shape[:-2], -1, self.attention_dim)
        if self.use_prosody:
            prosody_info = self.prosody_predictor(encoder_out, mel_mask, **prosody_target)
            encoder_out = prosody_info['output']
        
        if self.mel_before_conformer:
            melspec_out_chunked = self.attention2mel(encoder_out)  # (bs, mellen, 80)
        encoder_out = self.conformer_encoder(encoder_out, mel_mask)
        if not self.mel_before_conformer:
            melspec_out_chunked = self.attention2mel(encoder_out)  # (bs, mellen, 80)
        
        return {"visual_feature":feature,  # feature is still (bs, vidlen, 768)
                "melspec_out":melspec_out_chunked,  # (bs, mellen, 80)
                "prosody": prosody_info if self.use_prosody else None,
                "unit": unit_info if self.use_hubert_units else None,
                "hu": hu_info if self.use_hubert_representation else None,
                "output": encoder_out,  # (bs, mellen, attention_dim)
                }  

    