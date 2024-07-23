import math
import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import torch.nn.functional as F
import torch


class EERMetric:
    def __init__(self):
        self.labels = []
        self.predictions = []
        self.nan_check = True  # disable at your own risk
        
    def _torch2numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        else:
            return x
    
    def _check_nans(self, x, name):
        if any(math.isnan(x) for x in x): raise ValueError(f"NaN value found in {name}")

    def update(self, preds, labels):
        # Assuming preds and labels are numpy arrays when passed to this function.
        preds = self._torch2numpy(preds)
        labels = self._torch2numpy(labels)
        self.predictions.append(preds)
        self.labels.append(labels)

    def compute(self):
        # Flatten the lists and convert to numpy arrays
        all_preds = np.concatenate(self.predictions)
        all_labels = np.concatenate(self.labels)
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
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


if __name__ == '__main__':
    # Usage
    # Assume EERMetric class is already defined as previously provided
    metric = EERMetric()

    # Number of batches
    num_batches = 10
    batch_size = 7247  # Total of 72,474 pairs, distributed across 10 batches

    # Simulate random predictions and labels
    np.random.seed(42)  # For reproducibility
    for _ in range(num_batches):
        # Random predictions between 0 and 1
        preds = np.random.randint(-100, 100, size=(batch_size))/100
        # Random labels 0 or 1
        labels = np.random.randint(0, 2, size=batch_size)
        metric.update(preds, labels)

    # Compute the EER after all batches have been processed
    eer = metric.compute()
    print(f"Calculated EER: {eer:.4f}")

