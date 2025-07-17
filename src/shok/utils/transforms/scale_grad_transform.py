import torch

from shok.utils import functions


class ScaleGradTransform(torch.nn.Module):
    """Transforms scales the gradient of the input tensor."""

    def __init__(self):
        """Initialize the ScaleGradTransform."""
        super().__init__()

    def forward(self, x, y=None):
        """Scale the gradient of the input tensor."""
        return functions.ScaleGrad.apply(x), y
