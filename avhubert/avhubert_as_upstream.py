from avhubert.avhubert import AVHubertModel
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq.dataclass.configs import FairseqDataclass
from typing import Dict, List, Optional, Tuple
from omegaconf import MISSING, II

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


class FrontendWithEncoder(nn.Module):
    # Interface
    def avhubert_grad(self, enable:bool):
        pass


class AVHubertEncoder(FrontendWithEncoder):
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
    def __init__(self, cfg, num_mels, mel_mode:bool, with_conformer:bool, size="M") -> None:
        super().__init__()
        self.mel_mode = mel_mode  # mel_mode: DiVISe when set to true in verbose mode. ReVISE when set to false with direct return.
        self.with_conformer = with_conformer
        self.attention_dim = self.lookup_table[size]["attention_dim"]
        self.avhubert_model = AVHubertModel(cfg=cfg)
        self.avhubert2downstream = torch.nn.Linear(cfg.encoder_embed_dim, self.attention_dim*4)  # ratio=4
        if self.with_conformer:
            self.conformer_encoder = ConformerEncoder(size)
        if self.mel_mode:
            self.attention2mel = torch.nn.Linear(self.attention_dim, num_mels)
        
    def avhubert_grad(self, enable:bool):
        for _, param in self.avhubert_model.named_parameters():
            param.requires_grad = enable
    
    def _get_output(self, vis_feature, encoder_out):
        if self.mel_mode:
            # DiVISe
            # Upsampling handled in place
            melspec_out_chunked = self.attention2mel(encoder_out)  # (bs, mellen, num_mels)
            return {
                "visual_feature":vis_feature,  # feature is still (bs, vidlen, 768)
                "melspec_out":melspec_out_chunked,  # (bs, mellen, num_mels)
                "output": encoder_out,  # (bs, vidlen, attention_dim)
                }
        else:
            # ReVISE
            # Upsamping handled by transpose conv outside
            return encoder_out  # (bs, vidlen, attention_dim*4)
    
    def forward(self, source, mel_mask=None):
        # source should only include video
        encoder_out, feature, mask = self.avhubert_model.extract_finetune_with_feature(source)  # (bs, vidlen, 768)
        encoder_out = self.avhubert2downstream(encoder_out)  # (bs, vidlen, attention_dim*4)
        # (bs, vidlen, attention_dim*4) -> (bs, 4*vidlen, attention_dim)
        encoder_out = encoder_out.reshape(*encoder_out.shape[:-2], -1, self.attention_dim)
        if self.with_conformer:
            encoder_out = self.conformer_encoder(encoder_out, mel_mask)
        if not self.mel_mode:
            # ReVISE+Conformer is upsampled in unit upsampler outside.
            encoder_out = encoder_out.reshape(*encoder_out.shape[:-2], -1, self.attention_dim*4)
        
        return self._get_output(feature, encoder_out)


class SVTSModel(FrontendWithEncoder):

    def __init__(self, target_mel_dim, size="L", group_norm=False):
        super().__init__()

        # encoder can use batch or group-norm
        # decoder has a combination of layer (encoder layers) and batch-norm (conv module)
        # self.encoder = Encoder()
        self.target_mel_dim = target_mel_dim
        self.conformer_backbone = ConformerEncoder(size, use_speaker_encoder=True)
        self.attention_dim = self.conformer_backbone.lookup(size)["attention_dim"]

        # PROJECTION LAYER
        self.projection_layer = torch.nn.Linear(
            in_features=self.attention_dim,
            out_features=4*self.target_mel_dim,
        )

        if group_norm:
            # recursive function to convert batch-norm layers to group-norm for gradient accumulation training
            def convert_bn_layer(module):
                for name, l in module.named_children():
                    if any([isinstance(l, x) for x in [torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d]]):
                        setattr(module, name, torch.nn.GroupNorm(32, num_channels=l.num_features))
                    if len(list(l.children())) > 0:
                        convert_bn_layer(l)
            convert_bn_layer(self)


    def forward(self, speaker_vid, speaker_wav, max_sample_seconds=4):
        speaker_wav = speaker_wav[..., :max_sample_seconds*16000]
        encoded = self.conformer_backbone(speaker_vid, xa=speaker_wav)  # (N, Lin, attn_dim)
        encoded = self.projection_layer(encoded)  # (N, Lin, 4*self.target_mel_dim)

        # reshape x
        melspec_out_chunked = encoded.reshape(*encoded.shape[:-2], -1, self.target_mel_dim)  # (N, 4*Lin, self.target_mel_dim)
        return {
            "melspec_out": melspec_out_chunked,  # (bs, mellen, num_mels)
            "output": encoded,  # (bs, vidlen, attention_dim)
        }