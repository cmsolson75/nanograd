class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError