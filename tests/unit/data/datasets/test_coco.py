import fiftyone
from PIL import Image

from shok.data.datasets.coco import CocoDataModule
from shok.data.datasets.utils import FiftyOneTorchDataset


def test_fiftyone_torch_dataset():
    # Load a FiftyOne dataset
    fiftyone_dataset = fiftyone.zoo.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        max_samples=10,  # Limit to 10 samples for testing
    )

    # Create a FiftyOneTorchDataset instance
    dataset = FiftyOneTorchDataset(
        fiftyone_dataset=fiftyone_dataset,
        transforms=None,
        gt_field="ground_truth",
        classes=None,
    )

    # Check the length of the dataset
    assert len(dataset) == 10

    # Check the first item in the dataset
    item = dataset[0]
    assert isinstance(item, tuple)
    assert len(item) == 2  # (image, target)
    assert isinstance(item[0], Image.Image)  # image should be a PIL Image
    assert isinstance(item[1], dict)  # target should be a dictionary
    assert "boxes" in item[1]  # target should contain 'boxes'


def test_coco_data_module():
    # Create a CocoDataModule instance
    data_module = CocoDataModule(batch_size=2, sample_size=1)

    # Prepare the data (this will download the dataset if not already present)
    data_module.prepare_data()

    # Setup the data module
    data_module.setup()

    # Check the train and validation datasets
    assert len(data_module.train_dataset) > 0
    assert len(data_module.val_dataset) > 0

    # Check the class mapping
    assert data_module.idx_to_class is not None
    assert isinstance(data_module.idx_to_class, dict)
    assert len(data_module.idx_to_class) > 0
