from unittest.mock import MagicMock

import pytest
import torch
from torchvision.transforms import v2

from shok.patch.module import ObjectDetectionPatch
from shok.utils.transforms import (
    PassRound,
    ScaleApplyPatch,
)
from shok.utils.transforms.apply_patch import ApplyPatch


@pytest.fixture
def dummy_model():
    # Create a dummy model with parameters
    model = torch.nn.Linear(10, 2)
    return model


def test_init_defaults(dummy_model):
    module = ObjectDetectionPatch(model=dummy_model)
    # Patch shape
    assert tuple(module.patch.shape) == (3, 2048, 2048)
    # Patch is nn.Parameter and requires_grad
    assert isinstance(module.patch, torch.nn.Parameter)
    assert module.patch.requires_grad
    # Model parameters are frozen
    for param in module.model.parameters():
        assert param.requires_grad is False
    # Patch combiner is ApplyPatch
    assert isinstance(module.patch_combiner, ApplyPatch)
    # Patched image transforms is v2.Compose
    assert isinstance(module.patched_image_transforms, v2.Compose)
    # Val patch combiner is ScaleApplyPatch
    assert isinstance(module.val_patch_combiner, ScaleApplyPatch)
    # Val patched image transforms is v2.Compose
    assert isinstance(module.val_patched_image_transforms, v2.Compose)
    # Automatic optimization is False
    assert module.automatic_optimization is False
    # EOT samples default
    assert module.eot_samples == 1
    # eval_maps is torch.nn.ModuleList with two MeanAveragePrecision
    assert isinstance(module.eval_maps, torch.nn.ModuleList)
    assert len(module.eval_maps) == 2


def test_init_custom_args(dummy_model):
    patch_shape = (1, 10, 10)
    lr = 0.1
    clip_values = (10, 20)
    eot_samples = 5
    eot_rate = 2
    patch_combiner = MagicMock()
    val_patch_combiner = MagicMock()
    patched_image_transforms = v2.Identity()
    val_patched_image_transforms = v2.Compose([PassRound()])
    module = ObjectDetectionPatch(
        model=dummy_model,
        patch_shape=patch_shape,
        learning_rate=lr,
        clip_values=clip_values,
        eot_samples=eot_samples,
        eot_rate=eot_rate,
        patch_combiner=patch_combiner,
        val_patch_combiner=val_patch_combiner,
        patched_image_transforms=patched_image_transforms,
        val_patched_image_transforms=val_patched_image_transforms,
    )
    assert tuple(module.patch.shape) == patch_shape
    assert module.hparams.learning_rate == lr
    assert module.hparams.clip_values == clip_values
    assert module.eot_samples == eot_samples
    assert module.hparams.eot_rate == eot_rate
    assert module.patch_combiner is patch_combiner
    assert module.val_patch_combiner is val_patch_combiner
    assert isinstance(module.patched_image_transforms, v2.Compose)
    assert module.val_patched_image_transforms is val_patched_image_transforms


def test_patch_values_within_clip(dummy_model):
    clip_values = (5, 10)
    module = ObjectDetectionPatch(model=dummy_model, clip_values=clip_values)
    patch = module.patch.data
    assert torch.all(patch >= clip_values[0])
    assert torch.all(patch <= clip_values[1])
