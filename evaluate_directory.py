import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC
from torch.utils.tensorboard import SummaryWriter
import logging
import os
import sys
import argparse

import wandb
from utils import plot_spectrogram
from audio.eval_utils import MetricsEvaluater, MyWav2Vec2Processor
from dataset.meldataset import LogMelSpectrogram
import light_hf_proxy


# PATHs
LRS3_AUDIO_PATH = "/data1/yfliu/lrs3/test"
LRS3_TXT_PATH = LRS3_AUDIO_PATH
LRS3_TEST_COUNTS = 1321
LRS2_AUDIO_PATH = "/data1/yfliu/lrs2ondisk/mvlrs_v1/audio/main"
LRS2_TXT_PATH = "/data1/yfliu/lrs2ondisk/mvlrs_v1/main"
LRS2_TEST_COUNTS = 1243

# argparse

parser = argparse.ArgumentParser(description="Evaluate audio clips for ASR")
parser.add_argument("--eval_path", type=str, required=True, help="Path to the audio clips for evaluation")
parser.add_argument("--dataset_type", type=str, choices=["lrs2", "lrs3"], required=True, help="Type of dataset: lrs2 or lrs3")
parser.add_argument("--suffix", type=str, default="", help="Optional suffix (e.g. 'vc') to filter specific files like '_vc.wav'")
parser.add_argument("--runname", help="if specified, will apply wandb")
args = parser.parse_args()

# Variables
LENGTH_GAP_TOLERANCE = 1600  #  0.1s

steps = 0
err_tot = {}
metrics = {}

path_to_eval = args.eval_path  # path of to-evaluate audio clips
args.checkpoint_path = os.path.join(args.eval_path, os.pardir)
dataset_type = args.dataset_type  # lrs2 or lrs3
if dataset_type == 'lrs2':
    AUDIO_PATH_GT = LRS2_AUDIO_PATH
    TXT_PATH_GT = LRS2_TXT_PATH
    NUM_ALL_SAMPLES = LRS2_TEST_COUNTS
    AUDIO_POSTFIX = '.wav'
elif dataset_type == 'lrs3':
    AUDIO_PATH_GT = LRS3_AUDIO_PATH
    TXT_PATH_GT = LRS3_TXT_PATH
    NUM_ALL_SAMPLES = LRS3_TEST_COUNTS
    AUDIO_POSTFIX = '.flac'

# Logging
sw = SummaryWriter(os.path.join(args.checkpoint_path, 'logs'))
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logging.getLogger(__name__)
if args.runname:
    proj_name = 'Evaluation on '+ dataset_type.upper()
    wandb.init(project=proj_name, name=args.runname)

def log_final(err_tot):
    for err_key, err_term in err_tot.items():
        if err_key == 'algorithmic':
            continue
        if err_key not in err_tot['algorithmic']:
            err_term = err_term / (j+1)
        sw.add_scalar(f"test/{err_key}", err_term)
        metrics[err_key] = err_term
    if args.runname:
        wandb.log(metrics)
    print(metrics)

# Functions
def filter_audio_files(file_path, suffix=""):
    """ Filter files based on suffix and audio file extensions. """
    valid_extensions = ('.wav', '.flac', '.mp3')
    if not file_path.endswith(valid_extensions):
        return False
    if suffix and not any([file_path.endswith(f"_{suffix}{ext}") for ext in valid_extensions]):
        return False
    return True

def get_audio_files_recursive(directory, suffix=""):
    """ Recursively find audio files in the directory that match the suffix. """
    audio_files = []
    num_of_examples = 0
    for root, dirs, files in os.walk(directory):  # Recursively walk through directories
        for file in files:
            file_path = os.path.join(root, file)
            if filter_audio_files(file_path, suffix):  # Apply the filter function
                num_of_examples += 1
                audio_files.append(file_path)
    if num_of_examples < NUM_ALL_SAMPLES:
        logging.warning(f"{num_of_examples=} which is greater than {NUM_ALL_SAMPLES=}")
    elif num_of_examples < NUM_ALL_SAMPLES:     
        logging.warning(f"{num_of_examples=} which is less than {NUM_ALL_SAMPLES=}")    
    return audio_files

# Start
if torch.cuda.is_available():
    # Multi GPU is not currently supported.
    rank = 0
    device = torch.device('cuda:{:d}'.format(rank))
else:
    device = torch.device('cpu')
logging.info(f"running on {device=}")
logmel = LogMelSpectrogram().to(device)
w2v_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(device)
w2v_processor = MyWav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
audioevaluater = MetricsEvaluater(
    w2v_processor=w2v_processor, 
    w2v_model=w2v_model,
    device=device,
)

audio_files_to_eval = get_audio_files_recursive(args.eval_path, suffix=args.suffix)
for j, audio_path_to_eval in enumerate(tqdm(audio_files_to_eval)):  # iterate over given path_to_eval
    if not filter_audio_files(audio_path_to_eval, suffix=args.suffix):
        continue
    rel_path = os.path.relpath(audio_path_to_eval, path_to_eval)  # extract relative path like `0Fi83BHQsMA` for lrs3 and `6330311066473698535` for lrs2
    rel_path = rel_path.replace(".wav", "").replace(".mp3", "").replace(".flac", "")
    if args.suffix:
        rel_path = rel_path.replace(f"_{args.suffix}", "")
    audio_path_gt = os.path.join(AUDIO_PATH_GT, rel_path) + AUDIO_POSTFIX
    audio_to_eval = torchaudio.load(audio_path_to_eval)[0].to(device)
    audio_gt = torchaudio.load(audio_path_gt)[0].to(device)
    minlength = min(len(audio_to_eval[-1]), len(audio_gt[-1]))
    gap = max(len(audio_to_eval[-1]), len(audio_gt[-1])) - minlength
    assert gap <= LENGTH_GAP_TOLERANCE, f"gap is {gap} > {LENGTH_GAP_TOLERANCE=} in {j}th sample."
    # align in length
    audio_gt = audio_gt[..., :minlength]
    audio_to_eval = audio_to_eval[..., :minlength]
    mel_gt = logmel(audio_gt).to(device)
    mel_to_eval = logmel(audio_to_eval).to(device)
    wav_padding_mask = torch.ones_like(audio_to_eval).bool()
    
    # all audio_path_gt has a similar file under the same dir ending in .txt that is its text transcription, read it!
    gt_text_path = os.path.join(TXT_PATH_GT, rel_path) + ".txt"  # assuming text files have the same name but with .txt extension
    lns = open(gt_text_path).readlines()
    gt_texts = [lns[0].strip().split(':')[-1].strip().lower(), ]
    text_asr = audioevaluater.eval_metrics(audio_to_eval, audio_gt, ~wav_padding_mask, gt_texts)  # could be none
    if j <= 4:
        sw.add_text(f'y_text_{j}', gt_texts[0], steps)
        if text_asr is not None:
            sw.add_text(f'y_text_{j}', text_asr[0], steps)
        sw.add_figure(f'y_spec_{j}', plot_spectrogram(mel_gt[0].squeeze(0).cpu()), steps)
        logging.info(f"saved {j}")
        if mel_to_eval is not None:
            sw.add_figure(f'y_hat_mel_{j}', plot_spectrogram(mel_to_eval[0].squeeze(0).cpu().numpy()), steps)

del w2v_model, w2v_processor
err_tot = {**err_tot, **audioevaluater.err_tot}
log_final(err_tot)