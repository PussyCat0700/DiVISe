import numpy as np
import torch
from torch.utils.data import Dataset
from dataset.avdatasets import AVHubertDataset
from dataset.meldataset import load_wav
from speaker_encoder.audio import preprocess_wav

class ContrastivePairTestDataset(Dataset):
    def __getitem__(self, index):
        is_pos, item_a, item_b = self.pairs[index]
        return {
            "is_positive":is_pos,
            "item0":item_a,
            "item1":item_b,
            }

    
    def __len__(self):
        return len(self.pairs)

        
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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        pair_path = kwargs.pop("pair_path")
        self.pairs = self.read_pairs(pair_path)
        self.permute = kwargs.pop("permute", True)
        self.dataset = AVHubertDataset(*args, **kwargs)
    
    def __getitem__(self, index):
        videoitem = super().__getitem__(index)
        video_path0 = videoitem['item0']
        video_path1 = videoitem['item1']
        video0 = self.dataset.get_with_vidct(video_path0)
        video1 = self.dataset.get_with_vidct(video_path1)
        return videoitem['is_positive'], (video0, video1)  # [2, ...]
    
    def __len__(self):
        return len(self.dataset)

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
        if self.permute:
            video = collated['net_input']['source']['video']
            video = video.reshape(2, bs, *video.shape[1:])
            # [2, B, 1, T, H, W]
            # -> [B, T, 2, H, W, 1]
            collated['net_input']['source']['video'] \
                = video.permute(1, 3, 0, 4, 5, 2)
        # [B], collated
        return is_positive, collated
    

class AudioPairDataset(ContrastivePairTestDataset):
    def __init__(self, pair_path) -> None:
        super().__init__()
        self.pairs = self.read_pairs(pair_path)
        
    def __getitem__(self, index):
        audioitem = super().__getitem__(index)
        audio_path0 = audioitem['item0']
        audio_path1 = audioitem['item1']
        audio0 = preprocess_wav(audio_path0)
        audio1 = preprocess_wav(audio_path1)
        audio0 = torch.Tensor(audio0)
        audio1 = torch.Tensor(audio1)
        return audioitem['is_positive'], (audio0, audio1)  # [2, ...]
    
    def collater(self, batch):
        # Step 1: Split each tensor into two parts and organize them into two groups
        is_positive = np.array([tensor[0] for tensor in batch])
        group1 = [tensor[1][0] for tensor in batch]  # first elements of each pair
        group2 = [tensor[1][1] for tensor in batch]  # second elements of each pair
        # Step 2: Stack each group to create the new dimension for pairs
        groups = group1 + group2
        lengths = [len(x) for x in groups]
        collated = torch.nn.utils.rnn.pad_sequence(groups, batch_first=True)  # [B, Tmax]
        # Step 3: Update the mask to True for real logits
        padding_mask = torch.zeros_like(collated, dtype=torch.bool)
        for i, length in enumerate(lengths):
            padding_mask[i, :length] = True
        return is_positive, collated, padding_mask