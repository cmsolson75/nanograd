import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .tensor import Tensor


class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = max(0, int(num_workers))
        self.drop_last = drop_last
        self.indices = np.arange(len(dataset))
        self.current_idx = 0

        if shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_idx >= len(self.dataset):
            raise StopIteration

        batch_indices = self.indices[self.current_idx : self.current_idx + self.batch_size]
        if self.drop_last and len(batch_indices) < self.batch_size:
            raise StopIteration

        fetcher = self._fetch_parallel if self.num_workers > 0 else self._fetch_sequential
        batch_data = fetcher(batch_indices)
        self.current_idx += self.batch_size

        if not batch_data:
            raise StopIteration

        first_item = batch_data[0]
        if isinstance(first_item, (tuple, list)):
            components = list(zip(*batch_data))
            stacked = []
            for comp in components:
                dtype = comp[0].dtype if isinstance(comp[0], Tensor) else None
                stacked.append(self._stack(comp, dtype=dtype))
            return tuple(stacked) if len(stacked) > 1 else stacked[0]
        else:
            dtype = first_item.dtype if isinstance(first_item, Tensor) else None
            return self._stack(batch_data, dtype=dtype)

    def __len__(self):
        total = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size != 0:
            total += 1
        return total

    def _fetch_sequential(self, batch_indices):
        return [self.dataset[i] for i in batch_indices]

    def _fetch_parallel(self, batch_indices):
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            return list(executor.map(self.dataset.__getitem__, batch_indices))

    def _ensure_tensor(self, item, dtype=None):
        if isinstance(item, Tensor):
            return item
        return Tensor(item, dtype=dtype) if dtype is not None else Tensor(item)

    def _stack(self, items, dtype=None):
        tensors = [self._ensure_tensor(item, dtype=dtype) for item in items]
        return Tensor.stack(tensors, dtype=dtype if dtype is not None else np.float32)
