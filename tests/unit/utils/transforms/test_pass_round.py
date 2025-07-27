"""Tests for the PassRound transform."""

import torch

from shok.utils.transforms.pass_round import PassRound


def test_pass_round_single_tensor():
    """Test PassRound with a single tensor input."""
    transform = PassRound()

    # Test with float values that need rounding
    input_tensor = torch.tensor([1.2, 2.7, 3.1, -0.8, -2.3])
    result, y = transform(input_tensor)

    expected = torch.round(input_tensor)

    assert torch.equal(result, expected)
    assert y is None
    assert isinstance(result, torch.Tensor)


def test_pass_round_single_tensor_with_y():
    """Test PassRound with a single tensor and y parameter."""
    transform = PassRound()

    input_tensor = torch.tensor([1.2, 2.7, 3.1])
    y_value = "test_value"

    result, returned_y = transform(input_tensor, y_value)

    expected = torch.round(input_tensor)

    assert torch.equal(result, expected)
    assert returned_y == y_value
    assert isinstance(result, torch.Tensor)


def test_pass_round_list_of_tensors():
    """Test PassRound with a list of tensors input."""
    transform = PassRound()

    # Test with different sized tensors
    tensor_list = [torch.tensor([1.2, 2.7]), torch.tensor([3.1, 4.8, 5.3]), torch.tensor([-0.8, -2.3, 0.5, 1.9])]

    result_list, y = transform(tensor_list)

    # Check that we get a list back
    assert isinstance(result_list, list)
    assert len(result_list) == len(tensor_list)
    assert y is None

    # Check each tensor is rounded correctly
    for original, rounded in zip(tensor_list, result_list, strict=False):
        expected = torch.round(original)
        assert torch.equal(rounded, expected)
        assert isinstance(rounded, torch.Tensor)


def test_pass_round_list_of_tensors_with_y():
    """Test PassRound with a list of tensors and y parameter."""
    transform = PassRound()

    tensor_list = [torch.tensor([1.2, 2.7]), torch.tensor([3.1, 4.8])]
    y_value = {"key": "value"}

    result_list, returned_y = transform(tensor_list, y_value)

    # Check that we get a list back
    assert isinstance(result_list, list)
    assert len(result_list) == len(tensor_list)
    assert returned_y == y_value

    # Check each tensor is rounded correctly
    for original, rounded in zip(tensor_list, result_list, strict=False):
        expected = torch.round(original)
        assert torch.equal(rounded, expected)


def test_pass_round_empty_list():
    """Test PassRound with an empty list."""
    transform = PassRound()

    empty_list = []
    result_list, y = transform(empty_list)

    assert isinstance(result_list, list)
    assert len(result_list) == 0
    assert y is None


def test_pass_round_single_element_list():
    """Test PassRound with a list containing a single tensor."""
    transform = PassRound()

    tensor_list = [torch.tensor([1.2, 2.7, 3.1])]
    result_list, y = transform(tensor_list)

    assert isinstance(result_list, list)
    assert len(result_list) == 1
    assert y is None

    expected = torch.round(tensor_list[0])
    assert torch.equal(result_list[0], expected)


def test_pass_round_integer_tensors():
    """Test PassRound with tensors that are already integers."""
    transform = PassRound()

    # Single integer tensor
    int_tensor = torch.tensor([1, 2, 3, 4])
    result, y = transform(int_tensor)

    expected = torch.round(int_tensor)
    assert torch.equal(result, expected)

    # List of integer tensors
    int_list = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
    result_list, y = transform(int_list)

    for original, rounded in zip(int_list, result_list, strict=False):
        expected = torch.round(original)
        assert torch.equal(rounded, expected)
