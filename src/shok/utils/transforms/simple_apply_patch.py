import torch


class SimpleApplyPatch(torch.nn.Module):
    """
    Super simple patch applying transformation.

    This is used for debugging and testing purposes.
    """

    def __init__(self):
        """Initializes the instance and calls the parent class constructor."""
        super().__init__()

    def forward(self, x: torch.Tensor, patch: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Forwards the input tensor `x` through the transformation.

        Applies a patch to the input tensor `x` by replacing its leading channels and spatial dimensions
        with those from the `patch` tensor. Optionally returns a target tensor `y`.

        Args:
            x (torch.Tensor): The input tensor to be modified.
            patch (torch.Tensor): The patch tensor to be inserted into `x`.
            y (torch.Tensor, optional): An optional target tensor to be returned.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the modified input tensor
            and the optional target tensor `y`.

        """
        x_copy = x.clone()
        x_copy[..., : patch.shape[-2], : patch.shape[-1]] = patch
        return x_copy, y
