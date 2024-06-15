import os
import gzip
import numpy as np
import requests
from typing import Tuple
from tensor import Tensor

class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

class MNIST(Dataset):
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    def __init__(self, root: str = ".", train: bool = True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.download()
        if self.train:
            self.images = self.load_images(os.path.join(root, self.files["train_images"]))
            self.labels = self.load_labels(os.path.join(root, self.files["train_labels"]))
        else:
            self.images = self.load_images(os.path.join(root, self.files["test_images"]))
            self.labels = self.load_labels(os.path.join(root, self.files["test_labels"]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return Tensor(image), Tensor(label, dtype=np.int32)

    def download(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        for name, url in self.files.items():
            file_path = os.path.join(self.root, url)
            if not os.path.exists(file_path):
                print(f"Downloading {url}...")
                response = requests.get(self.base_url + url)
                with open(file_path, 'wb') as f:
                    print(f)
                    f.write(response.content)
                print(f"Downloaded {url}")

    @staticmethod
    def load_images(file_path: str) -> np.ndarray:
        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28 * 28).astype(np.float32) / 255.0

    @staticmethod
    def load_labels(file_path: str) -> np.ndarray:
        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data.astype(np.int32)
