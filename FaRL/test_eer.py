import torch
import clip
from tqdm import tqdm
from data.dataloading import FaRLPairDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from contrastive.metrics import EERMetric, calc_cosine_similarity


pair_path = "/data1/yfliu/voxceleb2/voxceleb2_pictestpairs.txt"
model_path = "/data1/yfliu/model/FaRL/FaRL-Base-Patch16-LAIONFace20M-ep16.pth"
device ="cuda" if torch.cuda.is_available() else "cpu"
bs = 256
runname = 'test'
ckpt_path = f'/data1/yfliu/outputs/farl/{runname}'


sw = SummaryWriter(os.path.join(ckpt_path, 'logs'))
model, preprocess = clip.load("ViT-B/16", device="cpu")
model = model.to(device)
farl_state=torch.load(model_path) # you can download from https://github.com/FacePerceiver/FaRL#pre-trained-backbones
model.load_state_dict(farl_state["state_dict"],strict=False)

dataset = FaRLPairDataset(pair_path)
dataloader = DataLoader(
    dataset,
    batch_size=bs,
    shuffle=False,
    num_workers=2,
)
eer_metric = EERMetric()
pbar = tqdm(dataloader)
for batch in pbar:
    with torch.no_grad():
        labels = batch[0].to(device)
        image_pair_a, image_pair_b = batch[1].to(device), batch[2].to(device)
        image_features_a = model.encode_image(image_pair_a)
        image_features_b = model.encode_image(image_pair_b)
        image_features = torch.stack((image_features_a, image_features_b))
        similarity = calc_cosine_similarity(image_features)
        eer_metric.update(similarity, labels)
final_eer = eer_metric.compute()
print(f'{final_eer=}')
sw.add_scalar(f"testing/eer", final_eer, 0)
