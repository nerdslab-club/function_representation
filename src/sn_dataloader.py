from torch.utils.data import Dataset, DataLoader


class SNDataloader:
    def __init__(self, dataset: Dataset, batch_size=8, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def getDataLoader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )
