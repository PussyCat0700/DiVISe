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
    QUANTIZATION = 'quantization'
    DIRECTMAPPING = 'directmapping'
    CLASSIFICATION = 'classification'
    EMBEDDING_METHODS = {QUANTIZATION, DIRECTMAPPING, CLASSIFICATION}
    # TODO: check pitch/energy min/max for LRS3
    def __init__(self, 
                 embedding_method:str,
                 pitch_min=None,
                 pitch_max=None,
                 energy_min=None,
                 energy_max=None,
                 n_bins=256, encoder_hidden=256,
                 ):
        super().__init__()
        assert embedding_method in self.EMBEDDING_METHODS
        self.embedding_method = embedding_method
        out_dim = n_bins if self.embedding_method == self.CLASSIFICATION else 1
        self.pitch_predictor = Predictor(in_dim=encoder_hidden, out_dim=out_dim)
        self.energy_predictor = Predictor(in_dim=encoder_hidden, out_dim=out_dim)
        if self.embedding_method != self.DIRECTMAPPING:
            # pitch_quantization ="log"
            pitch_quantization ="linear"  # TODO: linear should be enough? Verify.
            energy_quantization = "linear"
            n_bins = n_bins
            assert pitch_quantization in ["linear", "log"]
            assert energy_quantization in ["linear", "log"]
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
            self.pitch_embedding = nn.Embedding(
                n_bins, encoder_hidden
            )
            self.energy_embedding = nn.Embedding(
                n_bins, encoder_hidden
            )
        elif self.embedding_method == self.DIRECTMAPPING:
            # Technically these networks cannot be seen as "embedding"s in direct mapping mode.
            # We're calling them "embedding"s just for identical naming conventions.
            self.pitch_embedding = Predictor(in_dim=1, out_dim=encoder_hidden)
            self.energy_embedding = Predictor(in_dim=1, out_dim=encoder_hidden)
            
    def _get_embedding(self, x, target, mask, control, predictor, bins, embeddingparams):
        # control is not valid in classification mode.
        target_embedding_idx = None
        prediction = predictor(x, mask)
        if target is not None:
            target_embedding_idx = torch.bucketize(target, bins)
            embedding = embeddingparams(target_embedding_idx)
        else:
            if self.embedding_method == self.CLASSIFICATION:
                embedding_idx = prediction.max(dim=-1).indices
            else:
                prediction = prediction * control
                embedding_idx = torch.bucketize(prediction, bins)
            embedding = embeddingparams(
                embedding_idx
            )
        return prediction, embedding, target_embedding_idx

    def get_pitch_embedding(self, x, target, mask, control):
        return self._get_embedding(x, target, mask, control,
                                   predictor=self.pitch_predictor,
                                   bins=self.pitch_bins,
                                   embeddingparams=self.pitch_embedding,)
    
    def get_energy_embedding(self, x, target, mask, control):
        return self._get_embedding(x, target, mask, control,
                                   predictor=self.energy_predictor,
                                   bins=self.energy_bins,
                                   embeddingparams=self.energy_embedding,)
    
    def forward(
        self,
        x,
        mel_mask=None,
        pitch_target=None,  # definitely not used in directmapping embedding_method
        energy_target=None,  # definitely not used in directmapping embedding_method
        p_control=1.0,  # Invalid in classification mode
    ):
        if self.embedding_method != self.DIRECTMAPPING:
            pitch_prediction, pitch_embedding, pitch_target_embedding_idx = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            x = x + pitch_embedding
            
            energy_prediction, energy_embedding, energy_target_embedding_idx = self.get_energy_embedding(
                x, energy_target, mel_mask, p_control
            )
            x = x + energy_embedding
        elif self.embedding_method == self.DIRECTMAPPING:
            pitch_prediction = self.pitch_predictor(x, mel_mask)
            pitch_prediction = pitch_prediction * p_control
            pitch_embedding = self.pitch_embedding(pitch_prediction.unsqueeze(-1), mel_mask)
            x = x + pitch_embedding
            
            energy_prediction = self.energy_predictor(x, mel_mask)
            energy_prediction = energy_prediction * p_control
            energy_embedding = self.energy_embedding(energy_prediction.unsqueeze(-1), mel_mask)
            x = x + energy_embedding

        return {
            "output":x,
            "pitch_pred":pitch_prediction,
            "energy_pred":energy_prediction,
            "pitch_class": pitch_target_embedding_idx,
            "energy_class": energy_target_embedding_idx,
            "mel_mask":mel_mask,
        }