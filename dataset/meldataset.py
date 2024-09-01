import os
import torch
import torch.utils.data
import numpy as np
from librosa.filters import mel as librosa_mel_fn
import pyworld as pw
from scipy.interpolate import interp1d
import torchaudio.transforms as transforms
import torchaudio
import torch.nn.functional as F


def load_wav(full_path):
    data, sampling_rate = torchaudio.load(
        full_path
    )
    data = data.squeeze(0).numpy()
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = None
hann_window = None
def get_mel_basis(sampling_rate, n_fft, num_mels, fmin, fmax, device):
    mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel_basis = torch.from_numpy(mel).float().to(device)
    return mel_basis

def get_hann_window(win_size, device):
    hann_window = torch.hann_window(win_size).to(device)
    return hann_window

class LogMelSpectrogram(torch.nn.Module):
    def __init__(self, n_fft=1024, num_mels=128, hop_size=160, win_size=1024):
        super().__init__()
        self.n_fft=n_fft
        self.hop_size=hop_size
        self.melspctrogram = transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=self.n_fft,
            win_length=win_size,
            hop_length=self.hop_size,
            center=False,
            power=1.0,
            norm="slaney",
            onesided=True,
            n_mels=num_mels,
            mel_scale="slaney",
            f_max=8000,
        )

    def forward(self, wav):
        wav = F.pad(wav, ((self.n_fft-self.hop_size) // 2, (self.n_fft-self.hop_size) // 2), "reflect")
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel


# TODO update Griffin-Lim to match LogMelSpectrogram
class MelSpectrogramInverter(torch.nn.Module):
    def __init__(self, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, device) -> None:
        super().__init__()
        self.hop_length = hop_size
        self.n_fft = n_fft
        self.win_size = win_size
        mel_basis = get_mel_basis(sampling_rate, n_fft, num_mels, fmin, fmax, device)
        hann_window = get_hann_window(win_size, device)
        mel_basis_pseudoinv = torch.linalg.pinv(mel_basis)
        angles = torch.view_as_real(torch.exp(2j*torch.pi*torch.rand((5000))))  # to fit NCCL backend
        self.register_buffer('mel_basis', mel_basis_pseudoinv)
        self.register_buffer('angles', angles)
        self.register_buffer('hann_window', hann_window)
        
    def _move_buffer_to_device(self, device):
        self.mel_basis = self.mel_basis.to(device)
        self.angles = self.angles.to(device)
        self.hann_window = self.hann_window.to(device)
    
    def forward(self, mel_spectrogram, fit_length=True):
        """Converts mel spectrogram to waveform using torch
        
        Args:
            mel_spectrogram (torch.Tensor): torch.Tensor of shape (batch_size, mel_seq_length, num_mel): mel spectrogram input
            fit_length (bool, optional): If set to True, will pad the output to fit the hop_size upsamling rate perfectly. Defaults to True.

        Returns:
            torch.Tensor: (torch.Tensor): torch.Tensor of shape (batch_size, 1): reconstructed waveform
        """
        self._move_buffer_to_device(mel_spectrogram.device)
        D = spectral_de_normalize_torch(mel_spectrogram)
        inv_mid = D.transpose(-1, -2)
        S = self._mel_to_linear(inv_mid)  # Convert back to linear
        S = self._griffin_lim(S)
        if fit_length:
            S = torch.nn.functional.pad(S, (int(self.hop_length/2), int(self.hop_length/2)), mode='reflect')
        return S
    
    def _mel_to_linear(self, mel_spectrogram):
        lineared = torch.matmul(self.mel_basis, mel_spectrogram)
        lineared = torch.clamp(lineared, min=1e-10)
        return lineared
    
    def _istft(self, y):
        window = self.hann_window
        return torch.istft(y, self.n_fft, self.hop_length, self.win_size, window=window, return_complex=False)
    
    def _stft(self, y):
        window = self.hann_window
        return torch.stft(y, self.n_fft, self.hop_length, self.win_size, window=window, return_complex=True)
    
    def _griffin_lim(self, S):
        """librosa implementation of Griffin-Lim
        Based on https://github.com/librosa/librosa/issues/434
        """
        angles = torch.view_as_complex(self.angles[:S.shape[-1]]).unsqueeze(0)
        S_complex = torch.abs(S)
        if not torch.is_complex(angles):  # for compatibility
            angles = torch.view_as_complex(angles)
        y = self._istft(S_complex * angles)
        for _ in range(60):
            angles = torch.angle(self._stft(y))
            angles = torch.exp(1j * angles)
            y = self._istft(S_complex * angles)
        return y

# TODO: pitch should be computed prior to training. It will be a speed bottleneck otherwise.
# TODO: How about trying out for Kaldi Pitch (beta)? @ https://carolineechen.github.io/audio/main/tutorials/audio_feature_extractions_tutorial.html#kaldi-pitch-beta
def pitch(wav_batch:torch.Tensor, wav_padding_masks:torch.Tensor, sampling_rate=16000, hop_length=160, mode=None):
    assert mode in ['interpolate', None]
    wav_batch = wav_batch.squeeze().cpu().numpy()
    ret = []
    for (wav, padding_mask) in zip(wav_batch, wav_padding_masks):
        len_wav = sum(~padding_mask)
        wav = wav[:len_wav]
        wav = pitch_single(wav, mode, sampling_rate, hop_length)
        wav = torch.FloatTensor(wav)
        ret.append(wav)
    ret = torch.nn.utils.rnn.pad_sequence(ret, batch_first=True)
    return ret

def pitch_single(wav, mode, sampling_rate=16000, hop_length=160):
    # See FastSpeech 2's preprocessor.py
    pitch, t = pw.dio(
        wav.astype(np.float64),
        sampling_rate,  # 16000
        frame_period=hop_length / sampling_rate * 1000,  # 160/16000*1000=10
    )
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sampling_rate)
    if mode is not None:
        if mode == 'interpolate':
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))
    return pitch

def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files
