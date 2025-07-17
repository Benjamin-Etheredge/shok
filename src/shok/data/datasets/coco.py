import os

import fiftyone
import torch
import torchvision
from lightning.pytorch import LightningDataModule
from torchvision.transforms import v2

from shok.utils.transforms import TargetInsurance


class CocoDataModule(LightningDataModule):
    """
    CocoDataModule is a PyTorch Lightning DataModule for loading and managing the COCO 2017 dataset for object detection tasks.

    This module provides functionality to:
    - Download and prepare the COCO 2017 validation dataset using FiftyOne's zoo loader.
    - Load images and annotations using torchvision's CocoDetection.
    - Apply a sequence of transforms for preprocessing, including image conversion, bounding box validation, and dtype conversion.
    - Split the dataset into training and validation subsets.
    - Construct mappings from category IDs to class names for downstream use.
    - Provide PyTorch DataLoaders for training, validation, and (optionally) testing.

        train_dataset_repeat (Optional): Repeated training dataset (not implemented).
        wandb_classes (Optional): List of class names for use with Weights & Biases.
        fiftyone_dataset: FiftyOne dataset object loaded in prepare_data.

    Methods:
        __init__(batch_size: int = 2, sample_size: int = 1): Initializes the data module with batch and sample sizes.
        prepare_data(): Downloads and prepares the COCO 2017 validation dataset using FiftyOne.
        setup(stage=None): Loads and preprocesses the dataset, splits into train/val, and constructs class mappings.
        train_dataloader(): Returns a DataLoader for the training dataset.
        val_dataloader(): Returns DataLoaders for both training and validation datasets for evaluation.
        test_dataloader(): Not implemented; raises NotImplementedError.

        - The module is designed for use with PyTorch Lightning.
        - Data loading and splitting logic is customizable via TODOs.
        - The test dataloader is not implemented.

    """

    def __init__(
        self,
        batch_size: int = 2,
        sample_size: int = 1,
    ) -> None:
        """
        Initializes the dataset class with specified batch and sample sizes.

        Args:
            batch_size (int, optional): Number of samples per batch. Defaults to 2.
            sample_size (int, optional): Number of samples to draw. Defaults to 1.

        Attributes:
            base_dataset: The base dataset object.
            train_dataset: The training dataset object.
            train_dataset_repeat: The repeated training dataset object.
            val_dataset: The validation dataset object.
            idx_to_class: Mapping from index to class label.
            wandb_classes: List of class names for use with Weights & Biases.

        """
        super().__init__()
        self.save_hyperparameters()
        self.base_dataset = None
        self.train_dataset = None
        self.train_dataset_repeat = None
        self.val_dataset = None
        self.idx_to_class = None
        self.wandb_classes = None

    def prepare_data(self):
        """
        Loads the COCO 2017 validation dataset using FiftyOne's zoo loader and assigns it to `self.fiftyone_dataset`.

        The dataset is loaded with detection labels.

        Returns:
            None

        """
        # TODO figure out how to use
        self.fiftyone_dataset = fiftyone.zoo.load_zoo_dataset(
            "coco-2017",
            split="validation",
            label_types=["detections"],
        )

    # Note: No need to download again if already present, as prepare_data is called only once per run.
    def setup(self, stage=None):
        """
        Sets up the COCO dataset for training and validation.

        Loads the COCO 2017 validation images and annotations using torchvision's CocoDetection.
        Applies a sequence of transforms to the dataset, including image conversion, target insurance,
        and dtype conversion. Wraps the dataset for compatibility with transforms v2 and splits it into
        training and validation subsets. Also constructs a mapping from category IDs to class names.

        Args:
            stage (str, optional): Stage of setup (e.g., 'fit', 'test'). Defaults to None.

        Attributes:
            base_dataset (torchvision.datasets.CocoDetection): The base COCO detection dataset.
            train_dataset (torch.utils.data.Dataset): Training subset of the dataset.
            val_dataset (torch.utils.data.Dataset): Validation subset of the dataset.
            idx_to_class (dict): Mapping from category IDs to class names.

        """
        # TODO implement data loading logic
        base_dir = fiftyone.core.dataset.get_default_dataset_dir("coco-2017")
        root = os.path.join(base_dir, "validation", "data")
        ann_file = os.path.join(base_dir, "validation", "labels.json")
        self.base_dataset = torchvision.datasets.CocoDetection(
            root=root,
            annFile=ann_file,
            transforms=v2.Compose(
                [
                    v2.ToImage(),
                    # v2.SanitizeBoundingBoxes(),  # Ensure bounding boxes are valid
                    TargetInsurance(),  # make sure boxes and labels are present
                    v2.ToDtype(torch.float32, scale=False),
                ]
            ),
        )
        dataset = self.base_dataset
        dataset = torchvision.datasets.wrap_dataset_for_transforms_v2(
            dataset,
            target_keys=["boxes", "labels"],
        )
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset,
            [0.8, 0.2],
        )
        self.idx_to_class = (
            {item["id"]: item["name"] for item in self.base_dataset.coco.cats.values()}
            if hasattr(self.base_dataset, "coco")
            else None
        )
        # self.wandb_classes = wandb.Classes(
        #     [
        #         {
        #             "name": self.idx_to_class[idx],
        #             "id": idx,
        #             # "color": wandb.utils.generate_color(idx),
        #         }
        #         for idx in self.idx_to_class.keys()
        #     ]
        # )
        # self.val_dataset = RepeatDataset(self.val_dataset, repeat=self.sample_size)

    # def get_class_name(self, label):
    #     return self.train_dataset.coco.cats[label]['name'] if hasattr(self.train_dataset, 'coco') and label in self.train_dataset.coco.cats else str(label)

    def train_dataloader(self):
        """
        Creates and returns a DataLoader for the training dataset.

        Returns:
            torch.utils.data.DataLoader: DataLoader configured for training.

        Notes:
            - Uses the training dataset (`self.train_dataset`).
            - Batch size is set from hyperparameters (`self.hparams.batch_size`).
            - Data is shuffled for training.
            - Number of worker processes is determined by available CPU cores (minimum 1, maximum 8).
            - Persistent workers and pinned memory are enabled for performance.
            - Uses a custom collate function to unpack batches.

        """
        # TODO implement train dataloader
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,  # don't need to shuffle, but just in case some other batching wants to be tested
            num_workers=max(1, min(8, os.cpu_count() // 2)),
            persistent_workers=True,
            pin_memory=True,
            collate_fn=lambda x: tuple(zip(*x, strict=True)),  # Unpack the dataset
        )

    def val_dataloader(self):
        """
        Creates and returns validation dataloaders for the training and validation datasets.

        Returns:
            dict: A dictionary containing two DataLoader objects:
                - "clean_train": DataLoader for the training dataset (self.train_dataset) with validation settings.
                - "val": DataLoader for the validation dataset (self.val_dataset).

        Both DataLoaders use the following settings:
            - batch_size: Defined by self.hparams.batch_size.
            - shuffle: False (no shuffling).
            - num_workers: Number of worker processes, set to at least 1 and at most 8, based on available CPU cores.
            - persistent_workers: True (workers are kept alive between epochs).
            - pin_memory: True (enables faster data transfer to CUDA-enabled GPUs).
            - collate_fn: Function to unpack the dataset samples (tuple(zip(*x))).

        """
        val1 = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=max(1, min(8, os.cpu_count() // 2)),
            persistent_workers=True,
            pin_memory=True,
            collate_fn=lambda x: tuple(zip(*x, strict=True)),  # Unpack the dataset
        )
        val2 = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=max(1, min(8, os.cpu_count() // 2)),
            persistent_workers=True,
            pin_memory=True,
            collate_fn=lambda x: tuple(zip(*x, strict=True)),  # Unpack the dataset
        )
        return {
            "clean_train": val1,
            "val": val2,
        }

    def test_dataloader(self):
        """
        Creates and returns a DataLoader for the test dataset.

        Returns:
            torch.utils.data.DataLoader: DataLoader instance for the test dataset.

        Note:
            This method is not yet implemented.

        """
        # TODO implement test dataloader
        raise NotImplementedError("Test dataloader is not implemented yet.")
