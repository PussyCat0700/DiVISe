import torch
from torchaudio.functional import compute_kaldi_pitch

def pitch_single(wav, mode, sampling_rate=16000, hop_length=160):
    """
    Returns:
       Tensor: Pitch feature. Shape: ``(batch, frames 2)`` where the last dimension
       corresponds to pitch and NCCF.
    """
    wav = torch.FloatTensor(wav)
    # make up for frame_length
    frm_length = 25  # kaldi default frame_length
    frm_shift = hop_length / sampling_rate * 1000  # 160/16000*1000=10
    wav_offset = int(frm_length/frm_shift)*hop_length  # 3*160
    wav_offset = torch.zeros(*wav.shape[:-1], wav_offset)
    wav = torch.cat((wav, wav_offset), dim=-1)
    pitch_and_nfcc = compute_kaldi_pitch(
        waveform=wav,
        sample_rate=sampling_rate,  # 16000
        frame_shift=frm_shift,  # 10
    )
    return pitch_and_nfcc