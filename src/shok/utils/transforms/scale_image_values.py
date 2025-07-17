import torch


class ScaleImageValues(torch.nn.Module):
    """
    Simple transform scales the image values to be between 0 and 1.

    While the other v2 transforms do this, they seem to randomly mess with the labels.
    This transform ensures that the labels remain unchanged.
    """

    # this is used since the other transforms can mess with labels
    def __init__(self, min=0, max=255):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        """
        Scale the image values to be between 0 and 1.

        Args:
            x (torch.Tensor): Input image tensor.
            y (torch.Tensor, optional): Target tensor, not modified in this transform.

        Returns:
            torch.Tensor: Scaled image tensor.

        """
        return (x - self.min) / (self.max - self.min), y
