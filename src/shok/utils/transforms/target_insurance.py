import torch


class TargetInsurance(torch.nn.Module):
    """
    Transform that makes sure object detection targets are always present.

    Sometime the targets are not in the dataset and this breaks some torchvision transforms.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Ensures that the target dictionary `y` contains the keys "boxes" and "labels".

        If these keys are missing, initializes "boxes" with an empty tensor of shape (0, 4)
        and dtype float32, and "labels" with an empty tensor of dtype int64.

        Args:
            x (torch.Tensor): The input tensor.
            y (torch.Tensor): The target dictionary containing annotation data.

        Returns:
            Tuple[torch.Tensor, dict]: The input tensor and the updated target dictionary.

        """
        y["boxes"] = y.get("boxes", torch.zeros((0, 4), dtype=torch.float32))
        y["labels"] = y.get("labels", torch.zeros((0,), dtype=torch.int64))
        return x, y
