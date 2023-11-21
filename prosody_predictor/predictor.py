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
                 n_bins=256, encoder_hidden=256,
                 ):
        super().__init__()
        self.pitch_predictor = Predictor()
        self.energy_predictor = Predictor()

        self.pitch_feature_level = "frame_level"
        self.energy_feature_level = "frame_level"
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        # pitch_quantization ="log"
        pitch_quantization ="linear"  # TODO: linear should be enough? Verify.
        energy_quantization = "linear"
        n_bins = n_bins
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
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
        src_mask=None,
        mel_mask=None,
        pitch_target=None,
        energy_target=None,
        p_control=1.0,
    ):
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, p_control
            )
            x = x + energy_embedding

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, p_control
            )
            x = x + energy_embedding

        return {
            "output":x,
            "pitch_pred":pitch_prediction,
            "energy_pred":energy_prediction,
            "mel_mask":mel_mask,
        }