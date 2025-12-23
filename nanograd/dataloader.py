import numpy as np
import math
from .tensor import Tensor

"""
Need to add in workers to make this work better.
"""

class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
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
        
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        batch_data = [self.dataset[i] for i in batch_indices]
        data, targets = zip(*batch_data)
        
        self.current_idx += self.batch_size
        stacked_data = Tensor.stack(data)
        stacked_targets = Tensor.stack(targets, dtype=np.int32)
        return stacked_data, stacked_targets

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
