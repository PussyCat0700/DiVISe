import torch
from torchaudio.functional import compute_kaldi_pitch

def pitch_single(wav, mode, sampling_rate=16000, hop_length=160):
    """
    Returns:
       Tensor: Pitch feature. Shape: ``(batch, frames 2)`` where the last dimension
       corresponds to pitch and NCCF.
    """
    wav = torch.FloatTensor(wav)
    pitch_and_nfcc = compute_kaldi_pitch(
        waveform=wav,
        sample_rate=sampling_rate,  # 16000
        frame_shift=hop_length / sampling_rate * 1000,  # 160/16000*1000=10
    )
    return pitch_and_nfcc