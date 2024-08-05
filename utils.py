import glob
import os
import random
import tempfile
import cv2
import matplotlib
import numpy as np
import torch
from torch.nn.utils import weight_norm

from avhubert.avhubert_as_upstream import AVHubertPretrainingConfig
matplotlib.use("Agg")
import matplotlib.pylab as plt
from scipy.io.wavfile import write
from torch.optim.lr_scheduler import LRScheduler

class TriStageLRScheduler(LRScheduler):
    def __init__(self, optimizer, total_updates, t1, t2, last_lr_factor=0.05, last_epoch=-1, verbose=False):
        self.total_updates = total_updates
        self.t1 = t1
        self.t2 = t2
        self.t1_updates = int(self.total_updates * self.t1 / 100)
        self.t2_updates = int(self.total_updates * self.t2 / 100)
        self.t3_updates = self.total_updates - self.t1_updates - self.t2_updates
        self.last_lr_factor = last_lr_factor
        super(TriStageLRScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.t1_updates:
            # Linearly ramp up the learning rate
            return [base_lr * self.last_epoch / self.t1_updates for base_lr in self.base_lrs]
        elif self.last_epoch < self.t1_updates + self.t2_updates:
            # Keep the learning rate constant
            return [base_lr for base_lr in self.base_lrs]
        else:
            # Linearly decay the learning rate to last_lr_factor of the base_lr
            decay_progress = (self.last_epoch - self.t1_updates - self.t2_updates) / self.t3_updates
            return [base_lr - (base_lr * (1 - self.last_lr_factor) * decay_progress) for base_lr in self.base_lrs]

def seed_everything(seed):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class DataLoaderSeeder:
    def __init__(self, seed) -> None:
        self.seed = seed
    
    def __call__(self, k):
        return seed_everything(self.seed + (k * 10000))

def _basic_conv_variation(x, kernel_size, padding, stride):
    return torch.floor((x-kernel_size+2*padding)/stride+1)

def _get_mask(mask_lengths: torch.Tensor)->torch.Tensor:
    """get mask according to mask_lengths

    Args:
        mask_lengths (torch.Tensor): Tensor containing length of waveform

    Returns:
        torch.Tensor: shape (B, T'max) 
    """
    mask_lengths = mask_lengths.int().tolist()  # list of int
    output_mask = torch.nn.utils.rnn.pad_sequence([torch.ones(mask_length).bool() for mask_length in mask_lengths], batch_first=True)
    return output_mask

def mpd_length_variator(x:torch.Tensor, period:int):
    """length transform function from wavform to mpd outputs

    Args:
        x (torch.Tensor): a tensor containing length of waveform
        period (int): periods in MPD

    Returns:
        torch.Tensor: a tensor containing transformed lengths.
    """
    kernel_size=5
    stride=3
    assert period>0, f'{period=} but is expected to >0'
    # pad first
    x_mod_period = x%period
    x = x.where(x_mod_period==0, x+period-x_mod_period)
    assert (x%period == 0).all(), f'expected {x}%{period}==0, but got {x%period}'
    x = x // period

    for _ in range(4):
        padding = get_padding(5, 1)
        x = _basic_conv_variation(x, kernel_size=kernel_size, padding=padding, stride=stride)
    x *=period
    return x

def msd_length_variator(x:torch.Tensor):
    """length transform function from wavform to msd outputs

    Args:
        x (torch.Tensor): a tensor containing length of waveform

    Returns:
        torch.Tensor: a tensor containing transformed lengths.
    """
    # x is a tensor containing length of waveform
    kernels = [15, 41, 41, 41, 41, 41, 5]
    strides = [1, 2, 2, 4, 4, 1, 1]
    paddings = [7, 20, 20, 20, 20, 20, 2]
    for kernel, padding, stride in zip(kernels, paddings, strides):
        x = _basic_conv_variation(x, kernel_size=kernel, padding=padding, stride=stride)
    return x

def mpd_length_variators(in_mask:torch.Tensor):
    """transform in_mask into a series of loss masks for different layers of mpd

    Args:
        in_mask (torch.Tensor): padding mask for waveform
            It is a tensor of shape (..., max_length_in). Unmasked region should be filled with True.

    Returns:
        List[torch.Tensor]: a series of loss masks for different layers of mpd
    """
    ret_masks = []
    x = in_mask.sum(dim=-1)
    periods = [2, 3, 5, 7, 11]
    ys = [mpd_length_variator(x, period) for period in periods]
    for y in ys:
        y_mask = _get_mask(y).to(in_mask.device)
        ret_masks.append(y_mask)
    return ret_masks

def msd_length_variators(in_mask:torch.Tensor):
    """transform in_mask into a series of loss masks for different layers of msd

    Args:
        in_mask (torch.Tensor): padding mask for waveform
            It is a tensor of shape (..., max_length_in). Unmasked region should be filled with True.

    Returns:
        List[torch.Tensor]: a series of loss masks for different layers of msd
    """
    ret_masks = []
    x = in_mask.sum(dim=-1)
    for i in range(3):
        if i!=0:
            x = _basic_conv_variation(x, kernel_size=4, padding=2, stride=2)
        y = x
        y = msd_length_variator(y)
        y_mask = _get_mask(y).to(in_mask.device)
        ret_masks.append(y_mask)
    return ret_masks

def save_wav_16khz(wav_outdir:str, wav:torch.Tensor):
    """saves audio from item in AVHubertDataset

    Args:
        wav_name (str)
        wav (torch.Tensor): of size [T]
    """
    wav = wav.cpu().numpy()
    write(wav_outdir, 16_000, wav)
    
def denormalize_vidtensor(vid_tensor:torch.Tensor, image_mean:float, image_std:float):
    vid_tensor = vid_tensor.permute(1, 2, 3, 0).contiguous()  # [T, H, W, C]
    vid_tensor = vid_tensor*image_std+image_mean
    vid_tensor = vid_tensor*255+0.0
    return vid_tensor

def save_video(vid_outdir:str, vid_tensor:torch.Tensor, avhuberttaskconfig:AVHubertPretrainingConfig, ffmpeg_path:str):
    """saves video from item in AVHubertDataset

    Args:
        vid_name (str)
        vid_tensor (torch.Tensor): of size [C, T, H, W]
        image_mean (str): image_mean in avhubert config.
        image_std (str): img_std in avhubert config.
    """
    video = denormalize_vidtensor(vid_tensor, avhuberttaskconfig.image_mean, avhuberttaskconfig.image_std)
    video = video.cpu().numpy()
    with tempfile.TemporaryDirectory() as dirname:
        for i, img in enumerate(video):
            cv2.imwrite(os.path.join(dirname, str(i+1).zfill(4)+".png"), img)
        cmd = [ffmpeg_path, "-i", os.path.join(dirname,'%0'+str(4)+'d.png'), "-y", "-crf", "20", vid_outdir, "-loglevel", "quiet"]
        cmd = ' '.join(cmd)
        os.system(cmd)

def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def unwrap_module_generator(weight, ignore_conv_pre:bool):
    if ignore_conv_pre:
        return {k.replace("module.", ""): v for k, v in weight.items() if 'conv_pre' not in k}
    else:
        return {k.replace("module.", ""): v for k, v in weight.items()}

def unwrap_module_discriminator(weight, name):
    return {'.'.join(k.split('.')[2:]):v for k,v in weight.items() if name in k}

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix, load_best=False):
    if load_best:
        pattern = os.path.join(cp_dir, prefix + 'best')
        return pattern
    else:
        pattern = os.path.join(cp_dir, prefix + '[0-9]*')
        cp_list = glob.glob(pattern)
        if len(cp_list) == 0:
            return None
        return sorted(cp_list)[-1]

