# TODO random patch augs
# TODO disable grad for x? does that speed up training?
import torch
from torchvision import transforms as v2


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
        # size = rounded_size.tolist()
        # size = torch.round(torch.tensor(x_copy.shape[1:]) * patch_scale).to(torch.int32).tolist()
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

        # pad_top = torch.ceil(patch.shape[1] / 2).to(torch.int32).item()
        # pad_bottom = torch.floor(patch.shape[1] / 2).to(torch.int32).item()
        # pad_left = torch.ceil(patch.shape[2] / 2).to(torch.int32).item()

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

        if x_2 > x_copy.shape[-2]:
            raise ValueError(f"Patch exceeds image width: {x_2} > {x_copy.shape[-2]}")
        if y_2 > x_copy.shape[-1]:
            raise ValueError(f"Patch exceeds image height: {y_2} > {x_copy.shape[-1]}")

        # patch_x_1 = max(0, x_1)
        # height = y_2-y_1
        # width = x_2-x_1
        x_copy[..., x_1:x_2, y_1:y_2] = patch

        # crop back down
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
