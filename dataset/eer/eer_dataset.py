import numpy as np
from torch.utils.data import Dataset
from dataset.avdatasets import AVHubertDataset


class ContrastivePairTestDataset(Dataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        pair_path = kwargs.pop("pair_path")
        self.pairs = self.read_pairs(pair_path)
        self.dataset = AVHubertDataset(*args, **kwargs)
        
    
    def __getitem__(self, index):
        is_pos, item_a, item_b = self.pairs[index]
        return {
            "is_positive":is_pos,
            "item0":item_a,
            "item1":item_b,
            }

    
    def __len__(self):
        return len(self.dataset)

        
    def read_pairs(self, pair_path):
        pairs = []
        with open(pair_path, 'r') as f:
            lines = [x.strip() for x in f.readlines()]
            for line in lines:
                is_pos, item_a, item_b = line.split(' ')
                is_pos = int(is_pos)
                pairs.append((is_pos, item_a, item_b))
        return pairs
    

class VideoPairDataset(ContrastivePairTestDataset):    
    def __getitem__(self, index):
        videoitem = super().__getitem__(index)
        video_path0 = videoitem['item0']
        video_path1 = videoitem['item1']
        video0 = self.dataset.get_with_vidct(video_path0)
        video1 = self.dataset.get_with_vidct(video_path1)
        return videoitem['is_positive'], (video0, video1)  # [2, ...]
    

    def collater(self, batch):
        # Step 1: Split each tensor into two parts and organize them into two groups
        is_positive = np.array([tensor[0] for tensor in batch])
        group1 = [tensor[1][0] for tensor in batch]  # first elements of each pair
        group2 = [tensor[1][1] for tensor in batch]  # second elements of each pair
        # Step 2: Stack each group to create the new dimension for pairs
        groups = group1 + group2
        collated = self.dataset.collater(groups)
        # # You would need to combine these groups into a single tensor of shape [2, B] for your use
        bs = len(group1)
        collated = collated
        video = collated['net_input']['source']['video']
        video = video.reshape(2, bs, *video.shape[1:])
        # [2, B, 1, T, H, W]
        # -> [B, T, 2, H, W, 1]
        collated['net_input']['source']['video'] \
            = video.permute(1, 3, 0, 4, 5, 2)
        # [B], [B, T, 2, H, W, 1]
        return is_positive, collated