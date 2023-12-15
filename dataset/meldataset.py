import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import pyworld as pw
from scipy.interpolate import interp1d

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
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
    mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
    mel_basis = torch.from_numpy(mel).float().to(device)
    return mel_basis

def get_hann_window(win_size, device):
    hann_window = torch.hann_window(win_size).to(device)
    return hann_window

# TODO: Do we also need to apply wav padding mask before wav is transformed to mel-spectrogram?
def mel_spectrogram(*args, **kwargs):
    return mel_spectrogram_and_energy(*args, **kwargs)["spec"]

def mel_spectrogram_and_energy(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if mel_basis is None:
        mel_basis = get_mel_basis(sampling_rate, n_fft, num_mels, fmin, fmax, y.device)
    if hann_window is None:
        hann_window = get_hann_window(win_size, y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                      center=center, normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)  # to fit torch future features deprecating return_complex=False

    magnitude = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
    energy = torch.norm(magnitude, dim=-2)

    spec = torch.matmul(mel_basis, magnitude)
    spec = spectral_normalize_torch(spec)

    return {"spec":spec,
            "energy":energy,}

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


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size  # segment_size % hop_size == 0 must stands.
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename)
            audio = audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

            mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)["spec"]
        else:
            mel = np.load(
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)["spec"]

        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)
