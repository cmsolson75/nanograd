import numpy as np

from .tensor import Tensor


class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class TensorDataset(Dataset):
    """
    Simple dataset that wraps NumPy arrays or Python lists into Tensor pairs.
    """

    def __init__(self, data, targets, dtype=np.float32, target_dtype=np.int64):
        data = np.asarray(data, dtype=dtype)
        targets = np.asarray(targets, dtype=target_dtype)
        if len(data) != len(targets):
            raise ValueError("data and targets must have the same length")
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = Tensor(self.data[index], dtype=self.data.dtype)
        y = Tensor(self.targets[index], dtype=self.targets.dtype)
        return x, y


class GaussianMixtureDataset(Dataset):
    """
    Generates 2D samples from a small Gaussian mixture on demand.
    Designed to be thread-safe and reproducible when used with multiple workers.
    """

    def __init__(self, centers, noise=0.1, length=20000, seed=0):
        centers = np.asarray(centers, dtype=np.float32)
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError("centers must be of shape (num_components, 2)")
        self.centers = centers
        self.noise = noise
        self.length = length
        self.seed = seed

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + index)
        center_idx = rng.integers(0, len(self.centers))
        sample = self.centers[center_idx] + rng.normal(0, self.noise, size=2).astype(
            np.float32
        )
        return Tensor(sample, dtype=np.float32)
