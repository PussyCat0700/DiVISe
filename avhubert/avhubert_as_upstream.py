from omegaconf import OmegaConf
from avhubert.avhubert import AVHubertConfig, AVHubertModel
import torch
import torch.nn as nn

class AVHubertEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.avhubert_model = AVHubertModel(*args, **kwargs)
        self.linear_proj_layer = torch.nn.Linear(768, 320)
    
    def forward(self, xs):
        input = {"video": xs, "audio": None,}
        x, mask = self.avhubert_model.extract_finetune(input)  # (bs, inlen/4, attention_dim)
        x = self.linear_proj_layer(x)  # (bs, inlen/4, 320)
        return x

def load_avhubert_model(config_yaml_path):
    avhubert_config = OmegaConf.merge(AVHubertConfig(), OmegaConf.load(config_yaml_path))
    avhubert_visual_encoder = AVHubertEncoder(avhubert_config, [2004*[0]])
    return avhubert_visual_encoder
    