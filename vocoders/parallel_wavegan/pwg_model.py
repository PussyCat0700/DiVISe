import os
import yaml
from parallel_wavegan.utils import load_model
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

def load_config(checkpoint):
    dirname = os.path.dirname(checkpoint)
    config = os.path.join(dirname, "config.yml")
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.setdefault('normalize_before', False)
    return config

def load_pwg_model(checkpoint, config):
    model = load_model(checkpoint, config)
    if config['normalize_before']:
        assert hasattr(model, "mean"), "Feature stats are not registered."
        assert hasattr(model, "scale"), "Feature stats are not registered."
    model.remove_weight_norm()
    return model

class PWGModel(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        config = load_config(checkpoint)
        self.normalize_before = True  # Should be normalized according to pipeline of PWG
        self.model = load_pwg_model(checkpoint, config)
        self.register_buffer('logbase', torch.log(torch.Tensor([10])))
    
    def forward(self, x):
        batch = dict(normalize_before=self.normalize_before)
        ret=[]
        x = x/self.logbase  # change of base formula. PWG uses log10 in Mel-Spectrogram conversion.
        for single_item in x:
            batch.update(c=single_item)
            y = self.model.inference(**batch)
            ret.append(y)
        ret = pad_sequence(ret, batch_first=True,).squeeze(-1)
        return ret