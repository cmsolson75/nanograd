import numpy as np
import math

class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration

        start_index = self.current_index
        end_index = min(self.current_index + self.batch_size, len(self.indices))
        batch_indices = self.indices[start_index:end_index]
        batch = [self.dataset[i] for i in batch_indices]

        self.current_index = end_index
        return batch

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)