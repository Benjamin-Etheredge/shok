import torch

from shok.utils.functions.scale_grad import ScaleGrad


def test_forward_returns_input_tensor():
    input_tensor = torch.randn(5, 3, requires_grad=True)
    output = ScaleGrad.forward(None, input_tensor)
    assert torch.equal(output, input_tensor)


def test_forward_with_zeros():
    input_tensor = torch.zeros(4, requires_grad=True)
    output = ScaleGrad.forward(None, input_tensor)
    assert torch.equal(output, input_tensor)


def test_forward_with_integers():
    input_tensor = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
    output = ScaleGrad.forward(None, input_tensor)
    assert torch.equal(output, input_tensor)


def test_forward_preserves_dtype_and_shape():
    input_tensor = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    output = ScaleGrad.forward(None, input_tensor)
    assert output.shape == input_tensor.shape
    assert output.dtype == input_tensor.dtype
