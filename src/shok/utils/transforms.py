"""Custom torchvision transform operations designed for adversarial training."""

import torch

# from icecream import ic
from torchvision import transforms as v2
from torchvision.transforms import functional as F
from torchvision.tv_tensors import BoundingBoxes

from shok.utils import functions


# TODO random patch augs
# TODO disable grad for x? does that speed up training?
class ApplyPatch(torch.nn.Module):
    """Module to apply a patch to an image."""

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.1, 0.4),
        location_range: tuple[float, float] = (0.0, 1.0),
        patch_crop_range: tuple[float, float] = (0.8, 1.0),
        rotation_probs: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
        flip_probability: float = 0.5,
    ):
        """
        Initializes the transformation utility with configurable ranges and distributions for scaling, location, cropping, rotation, flipping, and color jitter.

        Args:
            scale_range (tuple[float, float], optional): Range for scaling patches. Defaults to (0.1, 0.4).
            location_range (tuple[float, float], optional): Range for selecting patch locations. Defaults to (0.0, 1.0).
            patch_crop_range (tuple[float, float], optional): Range for cropping patches. Defaults to (0.8, 1.0).
            rotation_probs (tuple[float, float, float, float], optional): Probabilities for selecting rotation angles. Defaults to (0.25, 0.25, 0.25, 0.25).
            flip_probability (float, optional): Probability of flipping the patch. Defaults to 0.5.

        Attributes:
            scale_range (tuple[float, float]): Range for scaling patches.
            location_range (tuple[float, float]): Range for selecting patch locations.
            location_distribution (torch.distributions.uniform.Uniform): Uniform distribution for patch location.
            patch_scale_distribution (torch.distributions.uniform.Uniform): Uniform distribution for patch scale.
            patch_crop_range (tuple[float, float]): Range for cropping patches.
            patch_crop_distribution (torch.distributions.uniform.Uniform): Uniform distribution for patch cropping.
            input_dims (tuple[int]): Input dimensions, default is (2,).
            rotation_distribution (torch.distributions.categorical.Categorical): Categorical distribution for rotation.
            flip_distribution (torch.distributions.bernoulli.Bernoulli): Bernoulli distribution for flipping.
            color_jitter (v2.ColorJitter): Color jitter transformation for brightness, contrast, saturation, and hue.

        """
        super().__init__()
        self.scale_range = scale_range
        # TODO adjust start end location with patch scale range
        # self.start_distribution = torch.distributions.half_normal.HalfNormal(
        #     loc=location_range[0], scale=(location_range[1] - location_range[0]) / 2
        # )

        # TODO change to half normal distribution
        self.location_range = location_range
        self.location_distribution = torch.distributions.uniform.Uniform(low=location_range[0], high=location_range[1])

        # TODO change to half normal distribution
        self.patch_scale_distribution = torch.distributions.uniform.Uniform(low=scale_range[0], high=scale_range[1])
        self.patch_crop_range = patch_crop_range
        self.patch_crop_distribution = torch.distributions.uniform.Uniform(
            low=patch_crop_range[0], high=patch_crop_range[1]
        )
        self.input_dims = (2,)  # NOTE could update to handle different shapes than images
        # self.rotation_distribution = torch.distributions.uniform.Uniform(
        #     low=rotation_probs[0], high=rotation_probs[1]
        # )
        # ic(rotation_probs)
        self.rotation_distribution = torch.distributions.categorical.Categorical(probs=torch.tensor(rotation_probs))
        self.flip_distribution = torch.distributions.bernoulli.Bernoulli(probs=flip_probability)
        # TODO use color jitter
        self.color_jitter = v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

    # TODO switch to crop then resize? use resized_crop?
    def forward(self, x: torch.Tensor, patch: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Forward method.

        The patch is randomly rotated, resized, and placed at a location determined by a distribution.
        The function ensures the patch fits within the image boundaries and updates the target tensor
        `y` if provided.

        Args:
            x (torch.Tensor): The input image tensor of shape (..., H, W).
            patch (torch.Tensor): The patch tensor to be inserted, typically of shape (..., h, w).
            y (torch.Tensor, optional): Target tensor containing annotations (e.g., bounding boxes and labels).
            Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - The transformed image tensor with the patch inserted.
                - The updated target tensor (if provided), otherwise None.

        """
        x_copy = x.clone()

        # NOTE do the rotation before computing and using the sizes
        patch = torch.rot90(
            patch,
            k=self.rotation_distribution.sample().item(),
            dims=(-2, -1),  # Rotate around the height and width dimensions
        )

        patch_scale = self.patch_scale_distribution.sample(self.input_dims)
        # TODO scale patch to a random size maybe? or keep to image ratio?
        # size = torch.round(torch.tensor(patch.shape[1:]) * patch_scale).to(torch.int32).tolist()
        scaled_shape = torch.tensor(x_copy.shape[-2:]) * patch_scale
        rounded_shape = torch.round(scaled_shape)
        rounded_size = rounded_shape.to(torch.int32)
        size = rounded_size
        # ic(size)
        # size = rounded_size.tolist()
        # size = torch.round(torch.tensor(x_copy.shape[1:]) * patch_scale).to(torch.int32).tolist()
        # ic(size)
        # patch = F.resize(
        #     patch,
        #     size=size,
        # )

        # TODO switch to functional resize to see if it fixes vmap
        resized_crop = v2.RandomResizedCrop(
            size=size,
            # scale=(self.scale_range[0], self.scale_range[1]),
            # ratio=self.patch_crop_range,
        )
        patch = resized_crop(patch)
        # ic(size)
        # ic(patch.shape)

        # pad_top = torch.ceil(patch.shape[1] / 2).to(torch.int32).item()
        # pad_bottom = torch.floor(patch.shape[1] / 2).to(torch.int32).item()
        # pad_left = torch.ceil(patch.shape[2] / 2).to(torch.int32).item()
        # ic(patch.shape, pad_top, pad_left)

        # pad out the image to allow patch to be placed at edges of image
        # left_right_pad = torch.tensor(patch.shape[1])
        # top_bottom_pad = torch.tensor(patch.shape[2])

        # left_pad = torch.ceil(left_right_pad).to(torch.int32).item()
        # top_pad = torch.ceil(top_bottom_pad).to(torch.int32).item()
        # # right_pad = 0
        # # bottom_pad = 0
        # right_pad = torch.floor(left_right_pad).to(torch.int32).item()
        # bottom_pad = torch.floor(top_bottom_pad).to(torch.int32).item()
        # top_bottom_pad = patch.shape[1]
        # # TODO if i'm already cropping the patch, does it make sense to pad the image for cropping?
        # left_right_pad = patch.shape[2]
        # x_copy = F.pad(
        #     x_copy,
        #     # padding=(left_pad, right_pad, top_pad, bottom_pad),
        #     # padding=(x.shape[2], x.shape[2], x.shape[1], x.shape[1]),
        #     # padding=(left_pad, top_pad, right_pad, bottom_pad),
        #     padding=(left_right_pad, top_bottom_pad)
        # )
        # ic((x_copy.shape[1] - x.shape[1]) / 2)
        # ic((x_copy.shape[2] - x.shape[2]) / 2)
        # assert x_copy.shape[1] - (patch.shape[1]*2) == x.shape[1], "something is off"
        # assert x_copy.shape[2] - (patch.shape[2]*2) == x.shape[2], "something is off"
        # ic("post padding x_copy.shape", x_copy.shape)

        # TODO update to be between like -1, 2 so the patch can start outside the image
        location_scale = self.location_distribution.sample(self.input_dims)

        # x_1, y_1 = torch.round(torch.tensor(x.shape[1:]) * location_scale).to(torch.int32)

        # TODO break up patch transforms to other transforms for more flexibility
        # for example make each transform not always used

        # the location doesn't make sense to put the patch at the right/bottom padded area
        # so we need to adjust the location
        # NOTE the patch is already cropped so we shouldn't worry about handling placing the patch off the edges
        # max_size = torch.tensor(x_copy.shape[-2:]) - torch.tensor(patch.shape[-2:])
        max_size = torch.tensor(x_copy.shape[-2:]) - torch.tensor(patch.shape[-2:])

        # xy_1 = max_size * location_scale
        x_1, y_1 = torch.round(max_size * location_scale).to(torch.int32)

        # patch_crop_scale = self.patch_crop_distribution.sample(self.input_dims)
        # patch_crop_x = torch.round(patch.shape[1] * patch_crop_scale[0]).to(torch.int32)
        # patch_crop_y = torch.round(patch.shape[2] * patch_crop_scale[1]).to(torch.int32)
        # left = patch_crop_y
        # top = patch_crop_x
        # height = min(patch.shape[1] - patch_crop_x, x_copy.shape[1] - x_1)
        # width = min(patch.shape[2] - patch_crop_y, x_copy.shape[2] - y_1)

        # patch = F.crop(
        #     patch,
        #     top=top,
        #     left=left,
        #     height=height,
        #     width=width,
        # )

        # TODO update to take any rotation
        # patch = F.rotate(
        #     patch,
        #     angle=self.rotation_distribution.sample(self.sample_size).item(),
        #     expand=True,  # Expand the image to fit the rotated patch
        # )

        # handle patch going off the edges of the image
        x_2 = x_1 + patch.shape[-2]
        y_2 = y_1 + patch.shape[-1]
        if x_2 <= x_copy.shape[-2]:
            raise ValueError("Patch exceeds image width")
        if y_2 <= x_copy.shape[-1]:
            raise ValueError("Patch exceeds image height")
        # patch_x_1 = max(0, x_1)
        # height = y_2-y_1
        # width = x_2-x_1
        x_copy[..., x_1:x_2, y_1:y_2] = patch

        # crop back down
        # TODO i've
        # x_copy = x_copy[:, top_bottom_pad:-top_bottom_pad, left_right_pad:-left_right_pad]
        # x_copy = x_copy[:, top_bottom_pad:-top_bottom_pad, left_right_pad:-left_right_pad]
        # x_copy = x_copy[:, left_pad:-right_pad, top_pad:-bottom_pad]

        # filter target boxes and labels
        # y_copy = y.copy() if y is not None else None
        # TODO adjust y?
        y_copy = y
        # if y_copy is not None:
        #     if "boxes" in y:
        #         # Adjust boxes to account for the patch location
        #         y_copy["boxes"][:, 0] = torch.clamp(y_copy["boxes"][:, 0] + x_1, min=0)
        #         y_copy["boxes"][:, 1] = torch.clamp(y_copy["boxes"][:, 1] + y_1, min=0)
        #         y_copy["boxes"][:, 2] = torch.clamp(y_copy["boxes"][:, 2] + x_1, max=x_copy.shape[1])
        #         y_copy["boxes"][:, 3] = torch.clamp(y_copy["boxes"][:, 3] + y_1, max=x_copy.shape[2])
        #     else:
        #         y_copy["boxes"] = torch.zeros((0, 4), dtype=torch.float32)

        #     if "labels" not in y_copy:
        #         y_copy["labels"] = torch.zeros((0,), dtype=torch.int64)

        return x_copy, y_copy


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


class ScaleApplyPatch(torch.nn.Module):
    """
    Applies a patch to an image at a scaled size.

    This is useful for evaluating patch effectiveness.
    """

    def __init__(self, scale=0.25):
        """
        Initializes the object with a specified scale factor.

        Args:
            scale (float, optional): The scale factor to be used. Defaults to 0.25.

        """
        super().__init__()
        self.scale = scale

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
        size = torch.round(torch.tensor(x.shape[1:]) * self.scale).to(torch.int32).tolist()
        patch = F.resize(patch, size=size)

        x_copy[:, : patch.shape[1], : patch.shape[2]] = patch

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
        x_copy[:, : patch.shape[1], : patch.shape[2]] = patch
        return x_copy, y


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


class SoftRound(torch.nn.Module):
    """
    Transform to use the soft round function for adversarial training.

    This is something being explored. Since rounding is not differentiable,
    additional logic is needed to ensure gradients can flow through the operation.

    This way does the rounding, but then calculates what the multiplier factor was.
    Then this value is used to scale the gradient.
    """

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        """
        Applies soft rounding to the input tensor using the SoftRound function.

        Args:
            x (torch.Tensor): Input tensor to be processed.
            y (optional): An optional secondary input, not used in the transformation.

        Returns:
            Tuple[torch.Tensor, Any]: A tuple containing the transformed tensor and the optional secondary input.

        """
        # Placeholder for soft rounding logic
        return functions.SoftRound.apply(x), y


class PassRound(torch.nn.Module):
    """
    A custom torch.nn.Module that applies a soft rounding operation to the input tensor.

    Args:
        x (torch.Tensor): The input tensor to be rounded.
        y (optional): An optional secondary input, passed through unchanged.

    Returns:
        Tuple[torch.Tensor, Any]: A tuple containing the rounded tensor and the optional secondary input.

    Note:
        The actual rounding logic is implemented in `functions.PassRound.apply`.

    """

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        """
        Applies a placeholder soft rounding operation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor to be processed.
            y (optional): Additional input, currently unused.

        Returns:
            Tuple[torch.Tensor, Any]: A tuple containing the processed tensor and the second input (y).

        """
        # Placeholder for soft rounding logic
        return functions.PassRound.apply(x), y


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


class ScaleGradTransform(torch.nn.Module):
    """Transforms scales the gradient of the input tensor."""

    def __init__(self):
        """Initialize the ScaleGradTransform."""
        super().__init__()

    def forward(self, x, y=None):
        """Scale the gradient of the input tensor."""
        return functions.ScaleGrad.apply(x), y


def default_patched_image_mutator():
    """Default image mutator for patching images."""
    return v2.Compose(
        [
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=(0, 360), expand=False, center=None, fill=None),
            v2.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # Translate by 10%
                scale=(0.9, 1.1),  # Scale by 10%
                shear=None,
                resample=False,
                fillcolor=None,
            ),
        ]
    )
