import fiftyone.utils.coco as fouc
import torch
from PIL import Image


class FiftyOneTorchDataset(torch.utils.data.Dataset):
    """
    A class to construct a PyTorch dataset from a FiftyOne dataset.

    Args:
        fiftyone_dataset: a FiftyOne dataset or view that will be used for training or testing
        transforms (None): a list of PyTorch transforms to apply to images and targets when loading
        gt_field ("ground_truth"): the name of the field in fiftyone_dataset that contains the
            desired labels to load
        classes (None): a list of class strings that are used to define the mapping between
            class names and indices. If None, it will use all classes present in the given fiftyone_dataset.

    """

    def __init__(
        self,
        fiftyone_dataset,
        transforms=None,
        gt_field="ground_truth",
        classes=None,
    ):
        """
        Initializes the dataset utility class.

        Args:
            fiftyone_dataset: A FiftyOne dataset or view containing the samples.
            transforms (callable, optional): Transformations to apply to the images and annotations. Defaults to None.
            gt_field (str, optional): The name of the ground truth field in the dataset. Defaults to "ground_truth".
            classes (list, optional): List of class labels. If None, distinct labels are extracted from the dataset. Defaults to None.

        Attributes:
            samples: The FiftyOne dataset or view.
            transforms: Transformations to apply to the data.
            gt_field: The ground truth field name.
            img_paths: List of image file paths from the dataset.
            classes: List of class labels, with "background" as the first class if not already present.
            labels_map_rev: Dictionary mapping class labels to their corresponding indices.

        """
        self.samples = fiftyone_dataset
        self.transforms = transforms
        self.gt_field = gt_field

        self.img_paths = self.samples.values("filepath")

        self.classes = classes
        if not self.classes:
            # Get list of distinct labels that exist in the view
            self.classes = self.samples.distinct(f"{gt_field}.detections.label")

        if self.classes[0] != "background":
            self.classes = ["background", *self.classes]

        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, idx):
        """
        Retrieves the image and its corresponding target annotations for the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple:
                - img (PIL.Image.Image): The loaded image in RGB format.
                - target (dict): Dictionary containing target annotations with the following keys:
                    - "boxes" (torch.Tensor): Bounding boxes in [x_min, y_min, x_max, y_max] format.
                    - "labels" (torch.Tensor): Class labels for each bounding box.
                    - "image_id" (torch.Tensor): Tensor containing the image index.
                    - "area" (torch.Tensor): Area of each bounding box.
                    - "iscrowd" (torch.Tensor): Crowd annotation for each bounding box.

        Notes:
            If transforms are defined, they are applied to both the image and target before returning.

        """
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        metadata = sample.metadata
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        area = []
        iscrowd = []
        detections = sample[self.gt_field].detections
        for det in detections:
            category_id = self.labels_map_rev[det.label]
            coco_obj = fouc.COCOObject.from_label(
                det,
                metadata,
                category_id=category_id,
            )
            x, y, w, h = coco_obj.bbox
            boxes.append([x, y, x + w, y + h])
            labels.append(coco_obj.category_id)
            area.append(coco_obj.area)
            iscrowd.append(coco_obj.iscrowd)

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.as_tensor([idx])
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        """
        Returns the number of image paths in the dataset.

        Returns:
            int: The total number of images.

        """
        return len(self.img_paths)

    def get_classes(self):
        """
        Returns the list of class labels associated with the dataset.

        Returns:
            list: A list containing the class labels.

        """
        return self.classes
