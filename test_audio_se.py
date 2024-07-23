import logging
import os
import sys
import time
from tqdm import tqdm
from dataset.eer.eer_dataset import AudioPairDataset
from contrastive.metrics import EERMetric
import speaker_encoder.inference as corentinJEncoder
from pathlib import Path
import torch
from torch.utils.data.dataloader import DataLoader
logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
logging.getLogger(__name__)


se_path = Path("/data1/yfliu/model/CorentinJ/encoder.pt")
pair_path = "/data1/yfliu/voxceleb2/voxceleb2_audiotestpairs.txt"
rank = 0
bs = 16


if __name__ == '__main__':
    torch.cuda.set_device(rank)  # A very strong boost. See https://github.com/jik876/hifi-gan/pull/25
    device = torch.device('cuda:{:d}'.format(rank))
    corentinJEncoder.load_model(se_path, device)
    eer_metric = EERMetric()
    audiopair_dataset = AudioPairDataset(pair_path=pair_path)
    dataloader = DataLoader(audiopair_dataset, batch_size=bs, collate_fn=audiopair_dataset.collater)
    pbar = tqdm(dataloader)
    logging.info("start")
    for batch in pbar:
        time_start = time.perf_counter()
        is_positive, audios, padding_mask = batch
        similarity = corentinJEncoder.compute_similarity(audios, padding_mask, max_audio_sample_size=4*16000, pad_audio=False)
        eer_metric.update(
                preds=similarity,
                labels=is_positive,
            )
        pbar.set_description(f'time past={time.perf_counter() - time_start}')
        logging.info(f"{is_positive.tolist()}, {similarity.tolist()}")
        try:
            curr_eer = eer_metric.compute()
            pbar.set_description(f'{curr_eer=}')
        except ValueError as e:
            pbar.set_description("got nan, continuing")
    logging.warn(eer_metric.compute())
        
    