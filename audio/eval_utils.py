import os
import editdistance
import nisqalib
import numpy as np
from pesq import pesq
from pystoi import stoi
import torch
from torchaudio.models.decoder import download_pretrained_files, ctc_decoder
import torchmetrics
from cypesq import NoUtterancesError
import speaker_encoder.inference as corentinJEncoder
import logging
from utils import save_wav_16khz
from librosa.feature import melspectrogram
from mel_cepstral_distance import get_metrics_mels


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
    total_mcd = get_mel_cepstral_distance(x, deg)
    
    return {
        "pesq": total_pesq,  # -0.5~4.5, higher the better
        "stoi": total_stoi,  # 0.0~1.0, higher the better
        "estoi": total_estoi,
        "mcd": total_mcd,
    }

class MetricsEvaluater:
    def __init__(self, w2v_processor:MyWav2Vec2Processor, w2v_model, device, postfix:str=None, num_classes:int=None) -> None:
        self.w2v_processor = w2v_processor
        self.w2v_model = w2v_model
        self.nisqa_model = nisqalib.NisqaModel("nisqa")
        self.postfix = postfix
        self.err_tot = {
            "stoi":0,
            "estoi":0,
            "pesq":0,
            "secs":0,
            "wer":0,
            "mcd":0,
            "mos_pred":0,
            "noi_pred":0,
            "dis_pred":0,
            "col_pred":0,
            "loud_pred":0,
        }   
        self.n_audio_metrics = 0
        self.wer_name = "wer"
        self.stoi_name = "stoi"
        self.estoi_name = "estoi"
        self.pesq_name = "pesq"
        self.secs_name = "secs"
        self.mcd_name = "mcd"
        self.nisqa_overall_name = "mos_pred"
        self.nisqa_noise_name = "noi_pred"
        self.nisqa_dis_name = "dis_pred"
        self.nisqa_col_name = "col_pred"
        self.nisqa_loud_name = "loud_pred"
        self.num_classes = num_classes
        self.unit_acc_name = "acc_hu_class"
        self.unit_recall_name = "recall_hu_class"
        self.unit_precision_name = "precision_hu_class"
        self.unit_auc_name = "auc_hu_class"
        self.n_err = 0
        self.n_total = 0
        if self.postfix is not None:
            self.err_tot = {f'{key}_{self.postfix}': 0 for key in self.err_tot}
            self.wer_name += f'_{self.postfix}'
            self.stoi_name += f'_{self.postfix}'
            self.estoi_name += f'_{self.postfix}'
            self.pesq_name += f'_{self.postfix}'
            self.secs_name += f'_{self.postfix}'
            self.mcd_name += f'_{self.postfix}'
            self.nisqa_overall_name += f'_{self.postfix}'
            self.nisqa_noise_name += f'_{self.postfix}'
            self.nisqa_dis_name += f'_{self.postfix}'
            self.nisqa_col_name += f'_{self.postfix}'
            self.nisqa_loud_name += f'_{self.postfix}'
        self.err_tot["algorithmic"] = set()
        if self.num_classes:
            task, average = "multiclass", "macro"
            # Unless you want to test with dataset loaded for ddp, don't remove sync_on_compute=False.
            # https://github.com/Lightning-AI/torchmetrics/pull/339
            self.acc = torchmetrics.Accuracy(task=task, num_classes=num_classes, average=average, sync_on_compute=False).to(device)
            self.recall = torchmetrics.Recall(task=task, num_classes=num_classes, average=average, sync_on_compute=False).to(device)
            self.precision = torchmetrics.Precision(task=task, num_classes=num_classes, average=average, sync_on_compute=False).to(device)
            self.auc = torchmetrics.AUROC(task=task, num_classes=num_classes, average=average, sync_on_compute=False).to(device)
            self.err_tot['algorithmic'].add(self.unit_acc_name)
            self.err_tot['algorithmic'].add(self.unit_recall_name)
            self.err_tot['algorithmic'].add(self.unit_precision_name)
            self.err_tot['algorithmic'].add(self.unit_auc_name)
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
    
    def compute_audio_metrics_torch(self, degs:torch.Tensor, refs:torch.Tensor, wav_padding_mask:torch.Tensor=None):
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
        nisqa_results = []
        for audio in degs:
            result = self.nisqa_model.predict(audio[None, ...], 16000)
            nisqa_results.append(result)
        rets = compute_audio_metrics_numpy(degs, refs, 16000)
        for i, ret in enumerate(rets):
            ret["secs"] = secs_list[i]
            for key, term in nisqa_results[i].items():
                ret[key] = term
        return rets
        
    def eval_metrics(self, g_hat, y, wav_padding_mask, gt_texts, preds_km=None, targets_km=None):      
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
            if self.num_classes:
                self.acc.update(preds_km, targets_km)
                self.recall.update(preds_km, targets_km)
                self.precision.update(preds_km, targets_km)
                self.auc.update(preds_km, targets_km)
                self.err_tot[self.unit_acc_name] = self.acc.compute().item()
                self.err_tot[self.unit_recall_name] = self.recall.compute().item()
                self.err_tot[self.unit_precision_name] = self.precision.compute().item()
                self.err_tot[self.unit_auc_name] = self.auc.compute().item()
            audio_metrics = self.compute_audio_metrics_torch(g_hat, y, ~wav_padding_mask)
            n_batch = len(audio_metrics)
            for audio_metric in audio_metrics:
                self.err_tot[self.stoi_name] += audio_metric["stoi"] / n_batch
                self.err_tot[self.estoi_name] += audio_metric["estoi"] / n_batch
                self.err_tot[self.pesq_name] += audio_metric["pesq"] / n_batch
                self.err_tot[self.secs_name] += audio_metric["secs"] / n_batch
                self.err_tot[self.mcd_name] += audio_metric["mcd"] / n_batch
                self.err_tot[self.nisqa_overall_name] += audio_metric["mos_pred"] / n_batch
                self.err_tot[self.nisqa_noise_name] += audio_metric["noi_pred"] / n_batch
                self.err_tot[self.nisqa_dis_name] += audio_metric["dis_pred"] / n_batch
                self.err_tot[self.nisqa_col_name] += audio_metric["col_pred"] / n_batch
                self.err_tot[self.nisqa_loud_name] += audio_metric["loud_pred"] / n_batch
            return hypoes


class SampleSaver:
    def __init__(self, outdir):
        self.saved_n_total = 0
        self.outdir = outdir

    @staticmethod
    def save_and_export(torch_item, input_dir_and_serial, output_dir_and_serial, specified_name):
        specified_name = output_dir_and_serial+"_"+specified_name
        save_wav_16khz(f"{specified_name}.mp3", torch_item)
        os.system(f"ffmpeg -y -i {input_dir_and_serial}.mp4 -i {specified_name}.mp3 -c:v copy -map 0:v:0 -map 1:a:0 -shortest {specified_name}.mp4>{specified_name}.log 2>&1")

    def __call__(self, wav_padding_mask, names, y_g_hat_vc=None, y_g_hat=None):
        for i, (padding_mask, name) in enumerate(zip(wav_padding_mask, names)):
            item_g_hat = None
            item_g_hat_vc = None
            input_dir_and_serial = ''.join(os.path.join(name["audio_basedir"], name["audio_id"]).split('audio'))
            rel_path = name['audio_basedir'].split('test')[-1].strip('/')
            output_dir_and_serial = os.path.join(self.outdir, rel_path)
            os.makedirs(output_dir_and_serial, exist_ok=True)
            output_dir_and_serial = os.path.join(output_dir_and_serial, name["audio_id"])
            if y_g_hat is not None:
                item_g_hat = y_g_hat[i].masked_select(~padding_mask)
                self.save_and_export(item_g_hat, input_dir_and_serial, output_dir_and_serial, "gf")
            if y_g_hat_vc is not None:
                item_g_hat_vc = y_g_hat_vc[i].masked_select(~padding_mask)
                self.save_and_export(item_g_hat_vc, input_dir_and_serial, output_dir_and_serial, "vc")
            self.saved_n_total += 1
        return self.saved_n_total


def get_mel_cepstral_distance(audio_1, audio_2, *, hop_length: int = 256, n_fft: int = 1024, window: str = 'hamming', center: bool = False, n_mels: int = 20, htk: bool = True, norm=None, dtype=np.float64, n_mfcc: int = 16, use_dtw: bool = True):
  """
  See get_metrics_wavs function in mel_cepstral_distance for docs.
  """

  sr_1 = sr_2 = 16_000

  mel_spectrogram1 = melspectrogram(
    y=audio_1,
    sr=sr_1,
    hop_length=hop_length,
    n_fft=n_fft,
    window=window,
    center=center,
    S=None,
    pad_mode="constant",
    power=2.0,
    win_length=None,
    # librosa.filters.mel arguments:
    n_mels=n_mels,
    htk=htk,
    norm=norm,
    dtype=dtype,
    fmin=0.0,
    fmax=None,
  )

  mel_spectrogram2 = melspectrogram(
    y=audio_2,
    sr=sr_2,
    hop_length=hop_length,
    n_fft=n_fft,
    window=window,
    center=center,
    S=None,
    pad_mode="constant",
    power=2.0,
    win_length=None,
    # librosa.filters.mel arguments:
    n_mels=n_mels,
    htk=htk,
    norm=norm,
    dtype=dtype,
    fmin=0.0,
    fmax=None,
  )

  mcd, penalty, _ = get_metrics_mels(mel_spectrogram1, mel_spectrogram2, n_mfcc=n_mfcc, take_log=True, use_dtw=use_dtw)
  return mcd+penalty