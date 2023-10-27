import glob
import os
import tempfile
import cv2
import matplotlib
import torch
from torch.nn.utils import weight_norm

from avhubert.avhubert_as_upstream import AVHubertPretrainingConfig
matplotlib.use("Agg")
import matplotlib.pylab as plt
from scipy.io.wavfile import write

def save_wav_16khz(wav_outdir:str, wav:torch.Tensor):
    """saves audio from item in AVHubertDataset

    Args:
        wav_name (str)
        wav (torch.Tensor): of size [T]
    """
    wav = wav.cpu().numpy()
    write(wav_outdir, 16_000, wav)
    
def save_video(vid_outdir:str, vid_tensor:torch.Tensor, avhuberttaskconfig:AVHubertPretrainingConfig, ffmpeg_path:str):
    """saves video from item in AVHubertDataset

    Args:
        vid_name (str)
        vid_tensor (torch.Tensor): of size [C, T, H, W]
        image_mean (str): image_mean in avhubert config.
        image_std (str): img_std in avhubert config.
    """
    vid_tensor = vid_tensor.permute(1, 2, 3, 0)  # [T, H, W, C]
    video = vid_tensor.cpu().numpy()
    video = video*avhuberttaskconfig.image_std+avhuberttaskconfig.image_mean
    video = video*255+0.0
    with tempfile.TemporaryDirectory() as dirname:
        for i, img in enumerate(video):
            cv2.imwrite(os.path.join(dirname, str(i+1).zfill(4)+".png"), img)
        import pdb
        pdb.set_trace()
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


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

