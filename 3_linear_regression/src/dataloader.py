import torch
from pshmodule.utils import filemanager as fm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split


class CustomDataset(Dataset):
    def __init__(
        self,
        data_path: str,
    ):
        data = fm.load(data_path)
        print(data.shape)
        print(data.head())
        
        self.X_train = torch.FloatTensor(data.iloc[:, :-1].to_numpy())
        self.y_train = torch.FloatTensor(data.iloc[:, -1].to_numpy())

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X_train[idx])
        y = torch.FloatTensor(self.y_train[idx])

        return x, y


class CustomDataLoader:
    def __init__(
        self,
        dataset,
        train_test_split: int,
        batch_size: int,
        shuffle_seed: int,
    ):
        self.dataset = dataset
        self.train_test_split = train_test_split
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed

    def dataloader(self):
        train_size = len(self.dataset) - int(len(self.dataset) * self.train_test_split)
        val_size = int(len(self.dataset) * self.train_test_split)

        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])

        train_dataset = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_seed,
        )
        val_dataset = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_seed,
        )

        return train_dataset, val_dataset
