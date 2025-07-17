import torch

from shok.utils.transforms.simple_apply_patch import SimpleApplyPatch


def test_forward_applies_patch_correctly():
    x = torch.zeros((1, 3, 8, 8))
    patch = torch.ones((1, 3, 4, 4))
    sap = SimpleApplyPatch()
    out, y = sap(x, patch)
    # Check that the top-left 4x4 region is replaced by ones
    assert torch.all(out[..., :4, :4] == 1)
    # Check that the rest remains zeros
    assert torch.all(out[..., 4:, :] == 0)
    assert torch.all(out[..., :, 4:] == 0)
    # y should be None by default
    assert y is None


def test_forward_with_y_argument():
    x = torch.zeros((2, 3, 8, 8))
    patch = torch.full((2, 3, 2, 2), 5.0)
    y = torch.tensor([1, 2])
    sap = SimpleApplyPatch()
    out, out_y = sap(x, patch, y)
    # Check patch region
    assert torch.all(out[..., :2, :2] == 5.0)
    # Check rest is zeros
    assert torch.all(out[..., 2:, :] == 0)
    assert torch.all(out[..., :, 2:] == 0)
    # y should be returned unchanged
    assert torch.equal(out_y, y)


def test_forward_patch_smaller_than_input():
    x = torch.randn((1, 3, 10, 10))
    patch = torch.randn((1, 3, 5, 5))
    sap = SimpleApplyPatch()
    out, _ = sap(x, patch)
    # Check that patch region matches patch
    assert torch.allclose(out[..., :5, :5], patch)
    # Check that outside patch region matches original x
    assert torch.allclose(out[..., 5:, :], x[..., 5:, :])
    assert torch.allclose(out[..., :, 5:], x[..., :, 5:])


def test_forward_patch_same_size_as_input():
    x = torch.randn((1, 3, 6, 6))
    patch = torch.randn((1, 3, 6, 6))
    sap = SimpleApplyPatch()
    out, _ = sap(x, patch)
    # Entire output should match patch
    assert torch.allclose(out, patch)
