import editdistance
import numpy as np
from pesq import pesq
from pystoi import stoi
import torch
from torchaudio.models.decoder import download_pretrained_files, ctc_decoder
from cypesq import NoUtterancesError
import logging
logger = logging.Logger(__name__)

class BeamSearchDecoder:
    def __init__(self, pretrained="librispeech-4-gram", lm_weight=3.23, word_score=-0.26):
        files = download_pretrained_files(pretrained)
        beam_search_decoder = ctc_decoder(
            lexicon=files.lexicon,
            tokens=files.tokens,
            lm=files.lm,
            nbest=3,
            beam_size=1500,
            lm_weight=lm_weight,
            word_score=word_score,
        )
        self.beam_search_decoder = beam_search_decoder
    
    def __call__(self, emission, lengths):
        beam_search_result = self.beam_search_decoder(emission.cpu().unsqueeze(0), lengths.cpu().unsqueeze(0))
        return beam_search_result

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor, length:int=None) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        if length is not None:
            emission = emission[:length]
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        raw_str = "".join([self.labels[i] for i in indices])
        return raw_str.replace("|", " ").lower().strip()

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

class AudioEvaluater:
    def __init__(self, transcriber, valid_greedy_decoder, err_tot, postfix=None) -> None:
        self.transcriber = transcriber
        self.valid_greedy_decoder = valid_greedy_decoder
        self.beamsearch = isinstance(self.valid_greedy_decoder, BeamSearchDecoder)
        self.postfix = postfix
        self.err_tot = err_tot
        self.n_audio_metrics = 0
        self.wer_name = "wer"
        self.stoi_name = "stoi"
        self.estoi_name = "estoi"
        self.pesq_name = "pesq"
        self.n_err = 0
        self.n_total = 0
        if self.postfix is not None:
            self.wer_name += f'_{self.postfix}'
            self.stoi_name += f'_{self.postfix}'
            self.estoi_name += f'_{self.postfix}'
            self.pesq_name += f'_{self.postfix}'
        # WER is computed with algorithmic averaging according to https://github.com/facebookresearch/av_hubert/blob/258fb50e155134eec2c4b49c2ae8de267075fd18/avhubert/infer_s2s.py#L254
        self.err_tot['algorithmic'].add(self.wer_name)
        
    def eval_metrics(self, g_hat, y, wav_padding_mask, gt_texts):      
        with torch.inference_mode():  
            wav_lengths = (~wav_padding_mask).sum(dim=-1)  # (batch_size,)
            # model definition can be found in https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/model.html
            emissions, lengths = self.transcriber(g_hat.squeeze(), wav_lengths)  # length indicates the valid length in time axis of emissions
            for emission, gt_text, length in zip(emissions, gt_texts, lengths):
                if self.beamsearch:
                    beam_search_result = self.valid_greedy_decoder(emission, length)
                    generated_text = " ".join(beam_search_result[0][0].words).strip()
                else:
                    generated_text = self.valid_greedy_decoder(emission, length)
                hypo, ref = generated_text.strip().split(), gt_text.strip().split()
                self.n_err += editdistance.eval(hypo, ref)
                self.n_total += len(ref)
            self.err_tot[self.wer_name] = self.n_err / self.n_total
            audio_metrics = compute_audio_metrics_torch(g_hat, y, 16000, ~wav_padding_mask)
            n_batch = len(audio_metrics)
            for audio_metric in audio_metrics:
                self.err_tot[self.stoi_name] += audio_metric["stoi"] / n_batch
                self.err_tot[self.estoi_name] += audio_metric["estoi"] / n_batch
                self.err_tot[self.pesq_name] += audio_metric["pesq"] / n_batch