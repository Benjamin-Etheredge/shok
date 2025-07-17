import torch

from shok.utils.functions.soft_round import SoftRound


class DummyCtx:
    def __init__(self):
        self.saved = None

    def save_for_backward(self, *args):
        self.saved = args


def test_forward_basic_rounding():
    ctx = DummyCtx()
    x = torch.tensor([0.2, 0.7, 1.5, -1.2, -2.8])
    result = SoftRound.forward(ctx, x)
    expected = torch.round(x)
    assert torch.allclose(result, expected)


def test_forward_requires_grad_delta_ratio():
    ctx = DummyCtx()
    x = torch.tensor([0.0, 1.2, -2.7, 0.0, 2.0], requires_grad=True)
    result = SoftRound.forward(ctx, x)
    expected = torch.round(x)
    assert torch.allclose(result, expected)
    # Check delta_ratio calculation
    rounded = expected
    delta_ratio = torch.where(rounded != 0, x / rounded, torch.zeros_like(x))
    delta_ratio = torch.where(torch.logical_and(x == 0, rounded == 0), torch.ones_like(x), delta_ratio)
    assert ctx.saved is not None
    saved_delta_ratio = ctx.saved[0]
    assert torch.allclose(saved_delta_ratio, delta_ratio)


def test_forward_zero_handling():
    ctx = DummyCtx()
    x = torch.tensor([0.0, 0.0], requires_grad=True)
    result = SoftRound.forward(ctx, x)
    expected = torch.round(x)
    assert torch.allclose(result, expected)
    delta_ratio = torch.ones_like(x)
    saved_delta_ratio = ctx.saved[0]
    assert torch.allclose(saved_delta_ratio, delta_ratio)


def test_forward_no_grad_no_ctx_saved():
    ctx = DummyCtx()
    x = torch.tensor([1.1, 2.9, -3.7])
    result = SoftRound.forward(ctx, x)
    expected = torch.round(x)
    assert torch.allclose(result, expected)
    assert ctx.saved is None
