import os
import torch
import torch.nn as nn
import numpy as np

from prosody_predictor.basic_blocks import Predictor

class ProsodyPredictor(nn.Module):
    """Implemented from Variance Adaptor in FastSpeech 2
    For reference, see
    https://github.com/ming024/FastSpeech2/blob/d4e79eb52e8b01d24703b2dfc0385544092958f3/model/modules.py#L17C19-L17C19
    """
    # TODO: check pitch/energy min/max for LRS3
    def __init__(self, pitch_min, pitch_max, energy_min, energy_max,
                 nccf_min=None, nccf_max=None,
                 n_bins=256, encoder_hidden=256,
                 ):
        super().__init__()
        self.pitch_predictor = Predictor()
        self.energy_predictor = Predictor()

        # pitch_quantization ="log"
        pitch_quantization ="linear"  # TODO: linear should be enough? Verify.
        energy_quantization = "linear"
        nccf_quantization = "linear"  # TODO: linear should be enough? Verify.
        n_bins = n_bins
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        self.kaldi_pitch = nccf_min is not None and nccf_max is not None
        def init_bin_params(minv, maxv, log_required):
            if log_required:
                params = torch.exp(
                    torch.linspace(np.log(minv), np.log(maxv), n_bins - 1)
                )
            else:
                params = torch.linspace(minv, maxv, n_bins - 1)
            return nn.Parameter(params, requires_grad=False)
        self.pitch_bins = init_bin_params(pitch_min, pitch_max, pitch_quantization == "log")
        self.energy_bins = init_bin_params(energy_min, energy_max, energy_quantization == "log")
        if self.kaldi_pitch:
            self.nccf_predictor = Predictor()
            self.nccf_bins = init_bin_params(nccf_min, nccf_max, nccf_quantization == "log")
            self.nccf_embedding = nn.Embedding(
                n_bins, encoder_hidden
            )
        self.pitch_embedding = nn.Embedding(
            n_bins, encoder_hidden
        )
        self.energy_embedding = nn.Embedding(
            n_bins, encoder_hidden
        )

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_nccf_embedding(self, x, target, mask, control):
        prediction = self.nccf_predictor(x, mask)
        if target is not None:
            embedding = self.nccf_embedding(torch.bucketize(target, self.nccf_bins))
        else:
            prediction = prediction * control
            embedding = self.nccf_embedding(
                torch.bucketize(prediction, self.nccf_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        x,
        mel_mask=None,
        pitch_target=None,
        energy_target=None,
        nccf_target=None,
        p_control=1.0,
    ):
        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            x, pitch_target, mel_mask, p_control
        )
        x = x + pitch_embedding
        
        if self.kaldi_pitch:
            nccf_prediction, nccf_embedding = self.get_nccf_embedding(
                x, nccf_target, mel_mask, p_control
            )
            x = x + nccf_embedding
            
        energy_prediction, energy_embedding = self.get_energy_embedding(
            x, energy_target, mel_mask, p_control
        )
        x = x + energy_embedding

        return {
            "output":x,
            "pitch_pred":pitch_prediction,
            "energy_pred":energy_prediction,
            "nccf_pred":nccf_prediction if self.kaldi_pitch else None,
            "mel_mask":mel_mask,
        }