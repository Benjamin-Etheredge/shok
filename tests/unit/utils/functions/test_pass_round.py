import torch

from shok.utils.functions.pass_round import PassRound


def test_forward_rounds_positive_floats():
    x = torch.tensor([0.1, 0.5, 1.4, 2.6, 3.9])
    result = PassRound.forward(None, x)
    expected = torch.round(x)
    assert torch.equal(result, expected)


def test_forward_rounds_negative_floats():
    x = torch.tensor([-0.1, -0.5, -1.4, -2.6, -3.9])
    result = PassRound.forward(None, x)
    expected = torch.round(x)
    assert torch.equal(result, expected)


def test_forward_rounds_mixed_values():
    x = torch.tensor([-2.7, -0.5, 0.0, 0.5, 2.7])
    result = PassRound.forward(None, x)
    expected = torch.round(x)
    assert torch.equal(result, expected)


def test_forward_with_integer_tensor():
    x = torch.tensor([1, 2, 3, 4])
    result = PassRound.forward(None, x)
    expected = torch.round(x)
    assert torch.equal(result, expected)


def test_forward_with_empty_tensor():
    x = torch.tensor([])
    result = PassRound.forward(None, x)
    expected = torch.round(x)
    assert torch.equal(result, expected)
