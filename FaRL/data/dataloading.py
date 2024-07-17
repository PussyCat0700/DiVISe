from torch.utils.data import Dataset
from PIL import Image
from .transform import default_transform


class BaseDataset(Dataset):
    def __init__(self, transform) -> None:
        super().__init__()
        self.transform = transform
        
    def preprocess(self, image_path):
        return self.transform(Image.open(image_path))


class FaRLPairDataset(BaseDataset):
    def __init__(self, pair_path, transform=default_transform) -> None:
        super().__init__(transform)
        self.pairs = self.read_pairs(pair_path)
        
    def read_pairs(self, pair_path):
        pairs = []
        with open(pair_path, 'r') as f:
            lines = [x.strip() for x in f.readlines()]
            for line in lines:
                is_pos, item_a, item_b = line.split(' ')
                is_pos = int(is_pos)
                pairs.append((is_pos, item_a, item_b))
        return pairs
    
    def __getitem__(self, index):
        is_pos, path_a, path_b = self.pairs[index]
        image_a = self.preprocess(path_a)
        image_b = self.preprocess(path_b)
        return is_pos, image_a, image_b
    
    def __len__(self):
        return len(self.pairs)


class FaRLTSVDataset(BaseDataset):
    def __init__(self, images, transform=default_transform) -> None:
        super().__init__(transform)
        self.files = self.read_lines(images)
        
    def read_lines(self, images):
        return [line.split('\t')[1] for line in images]
    
    def __getitem__(self, index):
        img_path = self.files[index]
        image = self.preprocess(img_path)
        return image
    
    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    try:
        from torchvision.transforms import InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC
    except ImportError:
        BICUBIC = Image.BICUBIC
    def _convert_image_to_rgb(image):
        return image.convert("RGB")
    n_px = 224
    transform = Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    dataset = FaRLPairDataset('/data1/yfliu/voxceleb2/voxceleb2_pictestpairs.txt', transform)
    print(dataset[0])
    loader = DataLoader(dataset, 4)
    for batch in loader:
        print(batch[0])