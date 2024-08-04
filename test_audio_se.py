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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
logging.getLogger(__name__)


se_path = Path("/data1/yfliu/model/CorentinJ/encoder.pt")
pair_path = "/data1/yfliu/voxceleb2/voxceleb2_audiotestpairs.txt"
bs = 16

def main(rank, world_size):
    # 初始化分布式环境
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method="tcp://localhost:52244")
    
    # 设置设备
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # 加载模型
    corentinJEncoder.load_model(se_path, device)
    
    # 创建 EERMetric 实例
    eer_metric = EERMetric(rank)
    
    # 数据加载
    audiopair_dataset = AudioPairDataset(pair_path=pair_path)
    sampler = DistributedSampler(audiopair_dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(audiopair_dataset, batch_size=bs, sampler=sampler, collate_fn=audiopair_dataset.collater)
    
    pbar = tqdm(dataloader, disable=not rank==0)
    logging.info("start")
    
    for batch in pbar:
        time_start = time.perf_counter()
        is_positive, audios, padding_mask = batch
        similarity = corentinJEncoder.compute_similarity(audios, padding_mask, max_audio_sample_size=4*16000, pad_audio=False)
        eer_metric.update(preds=similarity, labels=is_positive)
        
        pbar.set_description(f'time past={time.perf_counter() - time_start}')
    
    if rank == 0:
        # 计算并打印最终的 EER
        final_eer = eer_metric.compute()
        logging.warn(final_eer)
    
    # 清理
    dist.destroy_process_group()

if __name__ == '__main__':
    # 获取 rank 和 world_size 等参数
    world_size = torch.cuda.device_count()
    if world_size == 1:
        main(0, 1)
    else:
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
        
    