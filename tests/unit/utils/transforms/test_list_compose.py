"""Tests for the ListCompose class."""

import pytest
import torch
from torchvision.transforms import v2

from shok.utils.transforms.list_compose import ListCompose
from shok.utils.transforms.pass_round import PassRound
from shok.utils.transforms.scale_image_values import ScaleImageValues


def simple_add_one_transform(x, y=None):
    """Simple transform that adds 1 to all values."""
    if isinstance(x, list):
        return [tensor + 1 for tensor in x], y
    else:
        return x + 1, y


def simple_multiply_by_two(x):
    """Simple transform that multiplies by 2 (single argument)."""
    if isinstance(x, list):
        return [tensor * 2 for tensor in x]
    else:
        return x * 2


class TestListCompose:
    """Test cases for ListCompose."""

    def test_init_with_valid_transforms(self):
        """Test initialization with valid transforms."""
        transforms = [PassRound(), ScaleImageValues()]
        compose = ListCompose(transforms)

        assert len(compose) == 2
        assert isinstance(compose.transforms, torch.nn.ModuleList)

    def test_init_with_non_module_callable(self):
        """Test initialization with non-module callable."""
        transforms = [simple_add_one_transform, PassRound()]
        compose = ListCompose(transforms)

        assert len(compose) == 2
        # First should be wrapped, second should be as-is
        assert hasattr(compose.transforms[0], "transform")  # Wrapped
        assert isinstance(compose.transforms[1], PassRound)  # Direct

    def test_init_with_non_callable(self):
        """Test initialization fails with non-callable."""
        with pytest.raises(TypeError, match="is not callable"):
            ListCompose([PassRound(), "not_callable"])

    def test_init_with_non_iterable(self):
        """Test initialization fails with non-iterable."""
        # Use something that is truly not iterable
        with pytest.raises(TypeError, match="transforms should be an iterable"):
            ListCompose(123)  # Integer is not iterable

    def test_single_tensor_processing(self):
        """Test processing a single tensor through multiple transforms."""
        transforms = [simple_add_one_transform, simple_multiply_by_two]
        compose = ListCompose(transforms)

        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        result, y = compose(input_tensor)

        # Should be (input + 1) * 2 = [4.0, 6.0, 8.0]
        expected = torch.tensor([4.0, 6.0, 8.0])
        assert torch.equal(result, expected)
        assert y is None

    def test_single_tensor_with_y_parameter(self):
        """Test processing a single tensor with y parameter."""
        transforms = [simple_add_one_transform]
        compose = ListCompose(transforms)

        input_tensor = torch.tensor([1.0, 2.0])
        y_value = {"test": "data"}

        result, returned_y = compose(input_tensor, y_value)

        expected = torch.tensor([2.0, 3.0])
        assert torch.equal(result, expected)
        assert returned_y == y_value

    def test_list_of_tensors_processing(self):
        """Test processing a list of tensors."""
        transforms = [simple_add_one_transform, simple_multiply_by_two]
        compose = ListCompose(transforms)

        tensor_list = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0])]

        result_list, y = compose(tensor_list)

        # Each tensor should be (input + 1) * 2
        expected = [torch.tensor([4.0, 6.0]), torch.tensor([8.0, 10.0, 12.0])]

        assert len(result_list) == len(expected)
        for actual, exp in zip(result_list, expected, strict=False):
            assert torch.equal(actual, exp)
        assert y is None

    def test_list_of_tensors_with_y_parameter(self):
        """Test processing a list of tensors with y parameter."""
        transforms = [simple_add_one_transform]
        compose = ListCompose(transforms)

        tensor_list = [torch.tensor([1.0]), torch.tensor([2.0, 3.0])]
        y_value = "test_y"

        result_list, returned_y = compose(tensor_list, y_value)

        expected = [torch.tensor([2.0]), torch.tensor([3.0, 4.0])]

        assert len(result_list) == len(expected)
        for actual, exp in zip(result_list, expected, strict=False):
            assert torch.equal(actual, exp)
        assert returned_y == y_value

    def test_empty_list_processing(self):
        """Test processing an empty list."""
        transforms = [simple_add_one_transform]
        compose = ListCompose(transforms)

        result_list, y = compose([])

        assert isinstance(result_list, list)
        assert len(result_list) == 0
        assert y is None

    def test_with_real_transforms(self):
        """Test with actual transform modules."""
        transforms = [PassRound(), ScaleImageValues(min=0, max=1)]
        compose = ListCompose(transforms)

        # Test with single tensor
        input_tensor = torch.tensor([1.2, 2.7, 3.1])
        result, y = compose(input_tensor)

        # Should be rounded then scaled
        expected = torch.round(input_tensor)  # [1.0, 3.0, 3.0]
        expected = expected / 1.0  # No change since min=0, max=1

        assert torch.allclose(result, expected)
        assert y is None

    def test_with_real_transforms_list(self):
        """Test with actual transform modules on list input."""
        transforms = [PassRound(), ScaleImageValues(min=0, max=255)]
        compose = ListCompose(transforms)

        tensor_list = [torch.tensor([1.2, 2.7]), torch.tensor([3.1, 4.8, 5.3])]

        result_list, y = compose(tensor_list)

        # Each tensor should be rounded then scaled from [0, 255] to [0, 1]
        expected = [torch.round(tensor_list[0]) / 255.0, torch.round(tensor_list[1]) / 255.0]

        assert len(result_list) == len(expected)
        for actual, exp in zip(result_list, expected, strict=False):
            assert torch.allclose(actual, exp)

    def test_getitem(self):
        """Test accessing transforms by index."""
        pass_round = PassRound()
        scale_values = ScaleImageValues()
        transforms = [pass_round, scale_values]
        compose = ListCompose(transforms)

        assert compose[0] is pass_round
        assert compose[1] is scale_values

    def test_repr(self):
        """Test string representation."""
        transforms = [PassRound(), ScaleImageValues()]
        compose = ListCompose(transforms)

        repr_str = repr(compose)
        assert "ListCompose" in repr_str
        assert "PassRound" in repr_str
        assert "ScaleImageValues" in repr_str

    def test_callable_wrapper_single_arg_fallback(self):
        """Test that callable wrapper handles single-argument transforms."""

        def single_arg_transform(x):
            return x + 10

        transforms = [single_arg_transform]
        compose = ListCompose(transforms)

        input_tensor = torch.tensor([1.0, 2.0])
        result, y = compose(input_tensor, "test_y")

        expected = torch.tensor([11.0, 12.0])
        assert torch.equal(result, expected)
        assert y == "test_y"

    def test_callable_wrapper_invalid_signature(self):
        """Test that callable wrapper raises error for invalid signatures."""

        def invalid_transform():  # No arguments
            return torch.tensor([1.0])

        transforms = [invalid_transform]
        compose = ListCompose(transforms)

        input_tensor = torch.tensor([1.0, 2.0])

        with pytest.raises(TypeError, match="doesn't accept expected signature"):
            compose(input_tensor)

    def test_mixed_transforms(self):
        """Test mixing module transforms with callable functions."""

        def add_five(x, y=None):
            if isinstance(x, list):
                return [t + 5 for t in x], y
            return x + 5, y

        transforms = [
            add_five,  # Function
            PassRound(),  # Module
            simple_multiply_by_two,  # Single-arg function
        ]
        compose = ListCompose(transforms)

        input_tensor = torch.tensor([1.2, 2.7])
        result, y = compose(input_tensor)

        # (input + 5) -> round -> * 2
        # ([6.2, 7.7]) -> ([6.0, 8.0]) -> ([12.0, 16.0])
        expected = torch.tensor([12.0, 16.0])
        assert torch.equal(result, expected)

    def test_with_torchvision_transforms(self):
        """Test compatibility with standard torchvision transforms."""
        # Note: This test requires transforms that work with our (x, y) signature
        # Most torchvision transforms expect single tensor input

        def adapt_torchvision_transform(torchvision_transform):
            """Adapter to make torchvision transforms work with (x, y) signature."""

            def adapted_transform(x, y=None):
                if isinstance(x, list):
                    return [torchvision_transform(tensor) for tensor in x], y
                else:
                    return torchvision_transform(x), y

            return adapted_transform

        # Create adapted transforms
        normalize = adapt_torchvision_transform(v2.Normalize(mean=[0.5], std=[0.5]))

        transforms = [normalize]
        compose = ListCompose(transforms)

        # Test with single channel tensor
        input_tensor = torch.tensor([[0.8, 0.6], [0.4, 0.2]])  # 2x2 single channel
        input_tensor = input_tensor.unsqueeze(0)  # Add channel dimension

        result, y = compose(input_tensor)

        # Normalized: (x - 0.5) / 0.5
        expected = v2.Normalize(mean=[0.5], std=[0.5])(input_tensor)
        assert torch.allclose(result, expected)
        assert y is None
