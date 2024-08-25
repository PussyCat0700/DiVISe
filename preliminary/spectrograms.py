import os
import matplotlib.pylab as plt
import torch
import torchaudio.transforms as transforms
import torchaudio
import torch.nn.functional as F
import numpy as np


device = torch.device('cpu')


class MelSpectrogram(torch.nn.Module):
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
        )

    def forward(self, wav):
        wav = F.pad(wav, ((self.n_fft-self.hop_size) // 2, (self.n_fft-self.hop_size) // 2), "reflect")
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel


def plot_spectrogram(spectrogram, ax=None, cmap='viridis', savefile=None, fontsize=15):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none', cmap=cmap)
    
    # Increase font size for y-axis scale
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)  # Added for x-axis
    
    # Increase font size for colorbar legend
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=12)
    
    if savefile is not None:
        plt.tight_layout()
        plt.savefig(savefile)
    return ax


def highlight_correlated_areas(ground_truth, predicted, savepath, difference_threshold=1.0):
    # Compute correlation
    min_length = min(ground_truth.shape[-1], predicted.shape[-1])
    ground_truth = ground_truth[..., :min_length]
    predicted = predicted[..., :min_length]
    # Compute the absolute difference between the two spectrograms
    difference = np.abs(ground_truth - predicted)
    
    # Create a binary mask for highly correlated areas
    mask = np.zeros_like(predicted)
    mask[difference <= difference_threshold] = 1
    mask[ground_truth <= 0] = 0

    # Plot the predicted spectrogram
    fig, ax = plt.subplots(figsize=(10, 3))

    # First, plot the predicted spectrogram
    plot_spectrogram(predicted, ax=ax, cmap='viridis')

    # Overlay the binary mask as red contours
    ax.contour(mask, colors='red', levels=[0.5], linewidths=1.5)

    plt.tight_layout()
    plt.savefig(savepath)


def load_wav(full_path):
    data, sampling_rate = torchaudio.load(
        full_path
    )
    data = data.squeeze(0).numpy()
    return data, sampling_rate



mel = MelSpectrogram().to(device)
gts = {}
for root, dirs, files in os.walk('.'):
    for file_name in files:
        if file_name.endswith('.mp3') or file_name.endswith('.flac'):
            audio_fn = os.path.join(root, file_name)
            wav_data, sample_rate = load_wav(audio_fn)
            wav_data = torch.Tensor(wav_data).unsqueeze(0).to(device)
            spectrogram = mel(wav_data).squeeze().cpu().numpy()
            if 'gt' in root:
                gts[file_name] = spectrogram
            plot_spectrogram(spectrogram, savefile=audio_fn.replace('.mp3', '.png').replace('.flac', '.png'))

for root, dirs, files in os.walk('.'):
    for file_name in files:
        if file_name.endswith('.mp3') or file_name.endswith('.flac'):
            audio_fn = os.path.join(root, file_name)
            wav_data, sample_rate = load_wav(audio_fn)
            wav_data = torch.Tensor(wav_data).unsqueeze(0).to(device)
            spectrogram = mel(wav_data).squeeze().cpu().numpy()
            plot_spectrogram(spectrogram, savefile=audio_fn.replace('.mp3', '.png').replace('.flac', '.png'))
            # fig_dif = highlight_similarities(gts[file_name.replace('_', ' ').replace('.mp3', '.flac')], spectrogram)
            # fig_dif.savefig(audio_fn.replace('.mp3', '_diff.png').replace('.flac', '_diff.png'))
            fig_dif = highlight_correlated_areas(gts[file_name.replace('_', ' ').replace('.mp3', '.flac')], spectrogram, audio_fn.replace('.mp3', '_diff.png').replace('.flac', '_diff.png'))
