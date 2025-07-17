import torch
from torchvision.tv_tensors import BoundingBoxes


class ConvertToTVTensorBBoxes(torch.nn.Module):
    """
    Module to convert bounding boxes to torchvision tensors.

    This is a simplified version that does not include transformations.

    This is useful due to some torchvsion transforms requiring bounding boxes
    to be of type `torchvision.tv_tensors.BoundingBoxes`.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Applies transformation to input tensor `x` and optionally processes bounding boxes in `y`.

        Args:
            x (torch.Tensor): Input tensor, typically representing an image or batch of images.
            y (torch.Tensor, optional): Optional target dictionary. If provided and contains a "boxes" key,
                the bounding boxes are converted to a `BoundingBoxes` object in "xyxy" format with the same
                canvas size as `x` and dtype `torch.float32`.

        Returns:
            Tuple[torch.Tensor, dict]: The (possibly transformed) input tensor `x` and the
            updated target dictionary `y`.

        """
        if y is not None and "boxes" in y:
            y["boxes"] = BoundingBoxes(y["boxes"], format="xyxy", canvas_size=x.shape[1:], dtype=torch.float32)
        return x, y
