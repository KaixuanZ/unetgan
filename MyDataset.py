from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os

class Dataset(Dataset):
    def __init__(self, path):
        clean_names = lambda x: [i for i in x if i[0] != '.']
        self.imgpaths = [os.path.join(path, img) for img in clean_names(sorted(os.listdir(path)))]

    def __len__(self):
        return len(self.imgpaths)

    def __getitem__(self, index):
        img = Image.open(self.imgpaths[index])
        img = transforms.ToTensor()(img)
        img = (img-0.5)*2
        # import pdb; pdb.set_trace()
        return img


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img
