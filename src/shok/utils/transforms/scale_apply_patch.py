import torch
from torchvision.transforms import functional as F


class ScaleApplyPatch(torch.nn.Module):
    """
    Applies a patch to an image at a scaled size.

    This is useful for evaluating patch effectiveness.
    """

    def __init__(self, scale=0.25, preserve_aspect_ratio=True):
        """
        Initializes the object with a specified scale factor.

        Args:
            scale (float, optional): The scale factor to be used. Defaults to 0.25.
            preserve_aspect_ratio (bool, optional): Whether to preserve the aspect ratio during scaling. Defaults to True.

        """
        super().__init__()
        self.scale = scale
        self.preserve_aspect_ratio = preserve_aspect_ratio

    def forward(self, x: torch.Tensor, patch: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Applies a scaled patch to the input tensor `x` and optionally adjusts target annotations `y`.

        Args:
            x (torch.Tensor): The input tensor, typically an image of shape (C, H, W).
            patch (torch.Tensor): The patch tensor to be applied to `x`.
            y (torch.Tensor, optional): Target annotations dictionary containing keys such as "boxes" and "labels".

        Returns:
            Tuple[torch.Tensor, Optional[dict]]:
                - Modified input tensor with the patch applied.
                - Modified target annotations dictionary, if provided, with bounding boxes and labels
                adjusted to fit the new image dimensions.

        Notes:
            - The patch is resized according to a fixed scale before being applied.
            - Bounding boxes in `y` are clamped to ensure they remain within the image boundaries.
            - If "boxes" or "labels" are missing in `y`, they are initialized as empty tensors.

        """
        x_copy = x.clone()

        # Scale the patch to a fixed size
        if self.preserve_aspect_ratio:
            # TODO implement aspect ratio preservation
            pass

        size = torch.round(torch.tensor(x.shape[1:]) * self.scale).to(torch.int32).tolist()
        patch = F.resize(patch, size=size)

        x_copy[..., : patch.shape[-2], : patch.shape[-1]] = patch

        # TODO pull out or find built-in for this
        y_copy = y.copy() if y is not None else None
        if y_copy is not None:
            if "boxes" in y:
                # Adjust boxes to account for the patch location
                y_copy["boxes"][:, 0] = torch.clamp(y_copy["boxes"][:, 0], min=0)
                y_copy["boxes"][:, 1] = torch.clamp(y_copy["boxes"][:, 1], min=0)
                y_copy["boxes"][:, 2] = torch.clamp(y_copy["boxes"][:, 2], max=x_copy.shape[1])
                y_copy["boxes"][:, 3] = torch.clamp(y_copy["boxes"][:, 3], max=x_copy.shape[2])
            else:
                y_copy["boxes"] = torch.zeros((0, 4), dtype=torch.float32)

            if "labels" not in y_copy:
                y_copy["labels"] = torch.zeros((0,), dtype=torch.int64)

        return x_copy, y_copy
