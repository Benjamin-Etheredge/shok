import torch

from shok.utils.transforms.scale_grad_transform import ScaleGradTransform


class DummyScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # Just pass the gradient through unchanged for testing
        return grad_output


# Patch functions.ScaleGrad for testing
import shok.utils.functions  # noqa: E402

shok.utils.functions.ScaleGrad = DummyScaleGrad


def test_forward_returns_tuple():
    transform = ScaleGradTransform()
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.tensor([4.0, 5.0, 6.0])
    out, out_y = transform.forward(x, y)
    assert torch.allclose(out, x)
    assert torch.equal(out_y, y)


def test_forward_gradient_pass_through():
    transform = ScaleGradTransform()
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    out, _ = transform.forward(x)
    out.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_forward_with_none_y():
    transform = ScaleGradTransform()
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    out, out_y = transform.forward(x)
    assert torch.allclose(out, x)
    assert out_y is None
