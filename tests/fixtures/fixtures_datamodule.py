import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        """
        Returns the number of items in the dataset.

        Returns:
            int: The length of the dataset.

        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the data and label at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the data and corresponding label at the given index.

        """
        return self.data[idx], self.labels[idx]


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        """
        Sets up the datasets for training, validation, and testing stages.

        Args:
            stage (str, optional): The stage to set up. Defaults to None.

        Initializes:
            self.train_dataset: DummyDataset instance with 100 samples for training.
            self.val_dataset: DummyDataset instance with 20 samples for validation.
            self.test_dataset: DummyDataset instance with 20 samples for testing.

        """
        self.train_dataset = DummyDataset(size=100)
        self.val_dataset = DummyDataset(size=20)
        self.test_dataset = DummyDataset(size=20)

    def train_dataloader(self):
        """
        Returns a DataLoader for the training dataset.

        The DataLoader is initialized with the training dataset and the specified batch size.

        Returns:
            DataLoader: PyTorch DataLoader instance for the training dataset.

        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.

        The DataLoader is initialized with the validation dataset and the specified batch size.

        Returns:
            DataLoader: PyTorch DataLoader for the validation dataset.

        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        """
        Creates and returns a DataLoader for the test dataset.

        Returns:
            DataLoader: A DataLoader instance for iterating over the test dataset with the specified batch size.

        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


# Pytest fixture example


@pytest.fixture
def dummy_datamodule():
    dm = DummyDataModule(batch_size=16)
    dm.setup()
    return dm
