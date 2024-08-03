import editdistance
import numpy as np
from pesq import pesq
from pystoi import stoi
import torch
from torchaudio.models.decoder import download_pretrained_files, ctc_decoder
from cypesq import NoUtterancesError
import speaker_encoder.inference as corentinJEncoder
import logging
logger = logging.Logger(__name__)
# Torchaudio utils
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
# huggingface utils
from typing import Union, List
import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers import Wav2Vec2Processor
class MyWav2Vec2Processor(Wav2Vec2Processor):
    # Not using Huggingface's feature extractor
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
    
    def __call__(
        self,
        padded_inputs, 
        attention_mask,
    ) -> BatchFeature:
        padded_inputs = BatchFeature({"input_values": padded_inputs})
        if attention_mask is not None:
            padded_inputs["attention_mask"] = attention_mask

        # zero-mean and unit-variance normalization
        if self.feature_extractor.do_normalize:
            lengths = attention_mask.sum(dim=-1)
            padded_inputs["input_values"] = self.zero_mean_unit_var_norm(
                padded_inputs["input_values"], lengths=lengths, padding_value=self.feature_extractor.padding_value
            )

        padded_inputs["input_values"] = torch.stack(padded_inputs["input_values"])
        return padded_inputs
    @staticmethod
    def zero_mean_unit_var_norm(
        input_values: torch.Tensor, lengths:List[int], padding_value: float = 0.0
    ) -> List[torch.Tensor]:
        """
        Every array in the list is normalized to have zero mean and unit variance
        """
        if lengths is not None:
            normed_input_values = []

            for vector, length in zip(input_values, lengths):
                normed_slice = (vector - vector[:length].mean()) / torch.sqrt(vector[:length].var() + 1e-7)
                if length < normed_slice.shape[0]:
                    normed_slice[length:] = padding_value

                normed_input_values.append(normed_slice)
        else:
            normed_input_values = [(x - x.mean()) / torch.sqrt(x.var() + 1e-7) for x in input_values]

        return normed_input_values
# others
def compute_audio_metrics_torch(degs:torch.Tensor, refs:torch.Tensor, rate:int, wav_padding_mask:torch.Tensor=None):
    waveforms_stacked = torch.stack(
        [refs, degs],
        dim=0,
    )  # [2, B, T]
    waveforms_stacked = waveforms_stacked.reshape(-1, waveforms_stacked.shape[-1])
    wav_padding_mask_stacked = wav_padding_mask.repeat(2, 1)
    secs_list = corentinJEncoder.compute_similarity(
        waveforms_stacked,
        wav_padding_mask_stacked,
        max_audio_sample_size=4*16000,  # 4 seconds. Longer is better but consumes more mem.
        pad_audio=False,
    ).tolist()
    degs = [x.masked_select(mask).cpu().numpy() for mask, x in zip(wav_padding_mask, degs.squeeze().detach())]
    refs = [x.masked_select(mask).cpu().numpy() for mask, x in zip(wav_padding_mask, refs.squeeze().detach())]
    rets = compute_audio_metrics_numpy(degs, refs, rate)
    for i, ret in enumerate(rets):
        ret["secs"] = secs_list[i]
    return rets


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
    def __init__(self, w2v_processor:MyWav2Vec2Processor, w2v_model, err_tot, device, postfix=None) -> None:
        self.w2v_processor = w2v_processor
        self.w2v_model = w2v_model
        self.postfix = postfix
        self.err_tot = err_tot
        self.n_audio_metrics = 0
        self.wer_name = "wer"
        self.stoi_name = "stoi"
        self.estoi_name = "estoi"
        self.pesq_name = "pesq"
        self.secs_name = "secs"
        self.n_err = 0
        self.n_total = 0
        if self.postfix is not None:
            self.wer_name += f'_{self.postfix}'
            self.stoi_name += f'_{self.postfix}'
            self.estoi_name += f'_{self.postfix}'
            self.pesq_name += f'_{self.postfix}'
            self.secs_name += f'_{self.postfix}'
        # TODO magic path is bad
        from pathlib import Path
        se_path = Path("/data1/yfliu/model/CorentinJ/encoder.pt")
        corentinJEncoder.load_model(se_path, device)
        # WER is computed with algorithmic averaging according to https://github.com/facebookresearch/av_hubert/blob/258fb50e155134eec2c4b49c2ae8de267075fd18/avhubert/infer_s2s.py#L254
        self.err_tot['algorithmic'].add(self.wer_name)
        
    def map_to_pred(self, y, wav_padding_mask):
        device = y.device
        inputs = self.w2v_processor(y, ~wav_padding_mask)
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        with torch.no_grad():
            logits = self.w2v_model(input_values, attention_mask=attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.w2v_processor.batch_decode(predicted_ids)
        
        return transcription
        
    def eval_metrics(self, g_hat, y, wav_padding_mask, gt_texts):      
        with torch.inference_mode():  
            # model definition can be found in https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/model.html
            if len(g_hat.shape) > 2:
                g_hat = g_hat.squeeze(1)
            if len(y.shape) > 2:
                y = y.squeeze(1)
            assert g_hat.dim()==2 and y.dim()==2 and wav_padding_mask.dim()==2
            generated_texts = self.map_to_pred(g_hat, wav_padding_mask)  # length indicates the valid length in time axis of emissions
            hypoes = []
            for generated_text, gt_text in zip(generated_texts, gt_texts):
                generated_text = generated_text.lower().strip()
                hypo, ref = generated_text.strip().split(), gt_text.strip().split()
                self.n_err += editdistance.eval(hypo, ref)
                self.n_total += len(ref)
                hypoes.append(' '.join(hypo))
            self.err_tot[self.wer_name] = self.n_err / self.n_total
            audio_metrics = compute_audio_metrics_torch(g_hat, y, 16000, ~wav_padding_mask)
            n_batch = len(audio_metrics)
            for audio_metric in audio_metrics:
                self.err_tot[self.stoi_name] += audio_metric["stoi"] / n_batch
                self.err_tot[self.estoi_name] += audio_metric["estoi"] / n_batch
                self.err_tot[self.pesq_name] += audio_metric["pesq"] / n_batch
                self.err_tot[self.secs_name] += audio_metric["secs"] / n_batch
            return hypoes