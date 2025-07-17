import torch
from torchvision import transforms as v2

from shok.utils.transforms.apply_patch import ApplyPatch


def test_apply_patch_init_defaults():
    ap = ApplyPatch()
    assert ap.scale_range == (0.1, 0.4)
    assert ap.location_range == (0.0, 1.0)
    assert isinstance(ap.location_distribution, torch.distributions.uniform.Uniform)
    assert isinstance(ap.patch_scale_distribution, torch.distributions.uniform.Uniform)
    assert ap.patch_crop_range == (0.8, 1.0)
    assert isinstance(ap.patch_crop_distribution, torch.distributions.uniform.Uniform)
    assert ap.input_dims == (2,)
    assert isinstance(ap.rotation_distribution, torch.distributions.categorical.Categorical)
    assert isinstance(ap.flip_distribution, torch.distributions.bernoulli.Bernoulli)
    assert isinstance(ap.color_jitter, v2.ColorJitter)


def test_apply_patch_init_custom_values():
    scale_range = (0.2, 0.5)
    location_range = (0.1, 0.9)
    patch_crop_range = (0.7, 0.95)
    rotation_probs = (0.1, 0.2, 0.3, 0.4)
    flip_probability = 0.8
    ap = ApplyPatch(
        scale_range=scale_range,
        location_range=location_range,
        patch_crop_range=patch_crop_range,
        rotation_probs=rotation_probs,
        flip_probability=flip_probability,
    )
    assert ap.scale_range == scale_range
    assert ap.location_range == location_range
    assert ap.patch_crop_range == patch_crop_range
    assert torch.allclose(ap.location_distribution.low, torch.tensor(location_range[0]))
    assert torch.allclose(ap.location_distribution.high, torch.tensor(location_range[1]))
    assert torch.allclose(ap.patch_scale_distribution.low, torch.tensor(scale_range[0]))
    assert torch.allclose(ap.patch_scale_distribution.high, torch.tensor(scale_range[1]))
    assert torch.allclose(ap.patch_crop_distribution.low, torch.tensor(patch_crop_range[0]))
    assert torch.allclose(ap.patch_crop_distribution.high, torch.tensor(patch_crop_range[1]))
    assert torch.allclose(ap.rotation_distribution.probs, torch.tensor(rotation_probs))
    assert torch.allclose(ap.flip_distribution.probs, torch.tensor(flip_probability))


def test_apply_patch_color_jitter_params():
    ap = ApplyPatch()
    cj = ap.color_jitter
    # Check that the color jitter parameters are set as expected
    assert cj.brightness == (1 - 0.2, 1 + 0.2)
    assert cj.contrast == (1 - 0.2, 1 + 0.2)
    assert cj.saturation == (1 - 0.2, 1 + 0.2)
    assert cj.hue == (-0.1, 0.1)
