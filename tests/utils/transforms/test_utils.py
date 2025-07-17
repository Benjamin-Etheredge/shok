import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from shok.utils.transforms.utils import default_patched_image_mutator


def create_dummy_image(size=(64, 64), color=(128, 128, 128)):
    """Create a dummy PIL image for testing."""
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    return Image.fromarray(arr)


def test_default_patched_image_mutator_returns_compose():
    mutator = default_patched_image_mutator()
    assert isinstance(mutator, v2.Compose)
    assert len(mutator.transforms) == 5


def test_default_patched_image_mutator_transform_types():
    mutator = default_patched_image_mutator()
    types = [type(t) for t in mutator.transforms]
    expected_types = [
        v2.ColorJitter,
        v2.RandomHorizontalFlip,
        v2.RandomVerticalFlip,
        v2.RandomRotation,
        v2.RandomAffine,
    ]
    assert types == expected_types


def test_default_patched_image_mutator_applies_transforms():
    img = create_dummy_image()
    mutator = default_patched_image_mutator()
    transformed_img = mutator(img)
    assert isinstance(transformed_img, Image.Image)
    assert transformed_img.size == img.size  # Should remain the same size


def test_default_patched_image_mutator_is_deterministic_with_seed():
    img = create_dummy_image()
    mutator = default_patched_image_mutator()
    torch.manual_seed(42)
    out1 = mutator(img)
    torch.manual_seed(42)
    out2 = mutator(img)
    assert np.array_equal(np.array(out1), np.array(out2))
