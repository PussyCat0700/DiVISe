import math
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import torch.nn.functional as F
import torch
from torchmetrics.classification import BinaryROC
from torchmetrics.utilities.data import dim_zero_cat


class EERMetric(BinaryROC):
    higher_is_better = False
    def __init__(self, device):
        super().__init__()
        self.rank = device
        self.nan_check = True  # disable at your own risk
        self.to(self.rank)

    def update(self, preds: F.Tensor, labels: F.Tensor) -> None:
        preds = torch.Tensor(preds).to(self.rank)
        labels = torch.LongTensor(labels).to(self.rank)
        return super().update(preds, labels)
    
    def _check_nans(self, x, name):
        if any(math.isnan(x) for x in x): raise ValueError(f"NaN value found in {name}")
    
    def compute(self):
        # Taken from super().compute()
        # Torchmetric's _binary_clf_curve yields different results from what is given by sklearn
        # So we are manually implementing roc_curve here.
        state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)] if self.thresholds is None else self.confmat
        fpr, tpr, thresholds = roc_curve(state[1].cpu().numpy(), state[0].cpu().numpy())
        if self.nan_check:
            self._check_nans(fpr, "fpr")
            self._check_nans(tpr, "tpr")
        curve = lambda x: 1. - x - interp1d(fpr, tpr)(x)
        # Compute EER
        eer = brentq(curve, 0., 1.)
        return eer


def calc_cosine_similarity(embeddings):  
    # Assuming embeddings is your tensor of shape [2, B, C]
    # Compute cosine similarity between embeddings[0] and embeddings[1]
    cos_sim = F.cosine_similarity(embeddings[0], embeddings[1], dim=-1)
    return cos_sim


def single_gpu_test(preds_list, labels_list):
    # 单卡测试
    metric = EERMetric(torch.device('cuda:0'))
    
    # 使用提供的预测值和标签
    for preds, labels in zip(preds_list, labels_list):
        metric.update(torch.tensor(preds, dtype=torch.float32), labels)

    # 计算 EER
    eer = metric.compute()
    print(f"Calculated EER (Single GPU): {eer:.4f}")
    return eer

def multi_gpu_test(rank, world_size, preds_list1, preds_list2, labels_list1, labels_list2):
    if rank == 0:
        preds_list, labels_list = preds_list1, labels_list1
    else:
        preds_list, labels_list = preds_list2, labels_list2
    # 初始化分布式环境
    import torch.distributed as dist
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method='tcp://localhost:40742')
    
    # 创建 EERMetric 实例
    metric = EERMetric(rank)
    
    for preds, labels in zip(preds_list, labels_list):
        metric.update(torch.tensor(preds, dtype=torch.float32), labels)

    eer = metric.compute()
    print(f"Rank {rank}, Calculated EER: {eer:.4f}")
    
    # 清理
    dist.destroy_process_group()

if __name__ == '__main__':
    # 设置随机种子
    import random
    random.seed(42)
    np.random.seed(42)

    # 准备测试数据
    num_batches = 10
    batch_size = 7247  # 总共有 72,474 对
    preds_list1 = [np.random.randint(-100, 100, size=(batch_size)) / 100 for _ in range(num_batches//2)]
    labels_list1 = [np.random.randint(0, 2, size=batch_size) for _ in range(num_batches//2)]
    preds_list2 = [np.random.randint(-100, 100, size=(batch_size)) / 100 for _ in range(num_batches//2)]
    labels_list2 = [np.random.randint(0, 2, size=batch_size) for _ in range(num_batches//2)]

    # 单卡测试
    single_eer = single_gpu_test(preds_list1+preds_list2, labels_list1+labels_list2)
    
    # 多卡测试
    import torch.multiprocessing as mp
    world_size = torch.cuda.device_count()
    mp.spawn(multi_gpu_test, args=(world_size, preds_list1, preds_list2, labels_list1, labels_list2), nprocs=world_size, join=True)
    
    # 打印单卡和多卡结果对比
    print(f"Single GPU EER: {single_eer}")
    print("Ensure that all printed EER values from multi GPU match with the single GPU EER.")

