import numpy as np
from pesq import pesq
from pystoi import stoi
import torch
from cypesq import NoUtterancesError
import logging
logger = logging.Logger(__name__)
def compute_audio_metrics_torch(degs:torch.Tensor, refs:torch.Tensor, rate:int, wav_padding_mask:torch.Tensor=None):
    degs = [x.masked_select(mask).cpu().numpy() for mask, x in zip(wav_padding_mask, degs.squeeze().detach())]
    refs = [x.masked_select(mask).cpu().numpy() for mask, x in zip(wav_padding_mask, refs.squeeze().detach())]
    return compute_audio_metrics_numpy(degs, refs, rate)
def compute_audio_metrics_numpy(degs:np.array, refs:np.array, rate:int):
    """
    returns either a list of result or just one result depending on input.
    """
    if isinstance(degs, list) and isinstance(refs, list):
        rets = []
        for deg, ref in zip(degs, refs):
            try:
                ret = _compute_audio_metrics(deg, ref, rate)
                rets.append(ret)
            except NoUtterancesError as e:
                logging.warn('skipping one sample because no utterance was detected.')
        return rets
    else:
        return _compute_audio_metrics(degs, refs, rate)
def _compute_audio_metrics(deg, ref, rate):
    """
    input:
    deg: model generated audio (transferred from mel)
    ref: gt audio
    """
    x = ref
    total_pesq = pesq(16000, x, deg, 'nb')
    total_stoi = stoi(x, deg, rate, extended=False)
    total_estoi = stoi(x, deg, rate, extended=True)
    
    return {
        "pesq": total_pesq,  # -0.5~4.5, higher the better
        "stoi": total_stoi,  # 0.0~1.0, higher the better
        "estoi": total_estoi,
    }

