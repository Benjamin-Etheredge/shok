"""Tests for the ListTransformWrapper and BatchListTransformWrapper."""

import pytest
import torch
from torchvision import transforms

from shok.utils.transforms.list_wrapper import BatchListTransformWrapper, ListTransformWrapper, create_list_transform


class TestListTransformWrapper:
    """Test cases for ListTransformWrapper."""

    def test_initialization(self):
        """Test that wrapper initializes correctly."""
        transform = transforms.Normalize([0.5], [0.5])
        wrapper = ListTransformWrapper(transform)

        assert wrapper.transform == transform
        assert wrapper.skip_none is True
        assert wrapper.preserve_type is True

    def test_basic_transform_application(self):
        """Test basic transform application to list of tensors."""
        # Create a simple transform
        transform = transforms.Lambda(lambda x: x * 2)
        wrapper = ListTransformWrapper(transform)

        # Create test tensors of different sizes
        tensors = [
            torch.ones(2, 3),
            torch.ones(4, 5),
            torch.ones(1, 6),
        ]

        result = wrapper(tensors)

        assert len(result) == len(tensors)
        assert isinstance(result, list)

        # Check that transform was applied
        for i, tensor in enumerate(result):
            assert torch.allclose(tensor, tensors[i] * 2)

    def test_tuple_preservation(self):
        """Test that tuple type is preserved when preserve_type=True."""
        transform = transforms.Lambda(lambda x: x)
        wrapper = ListTransformWrapper(transform, preserve_type=True)

        tensors = (torch.ones(2, 2), torch.ones(3, 3))
        result = wrapper(tensors)

        assert isinstance(result, tuple)
        assert len(result) == len(tensors)

    def test_tuple_to_list_conversion(self):
        """Test tuple to list conversion when preserve_type=False."""
        transform = transforms.Lambda(lambda x: x)
        wrapper = ListTransformWrapper(transform, preserve_type=False)

        tensors = (torch.ones(2, 2), torch.ones(3, 3))
        result = wrapper(tensors)

        assert isinstance(result, list)
        assert len(result) == len(tensors)

    def test_none_handling_skip(self):
        """Test handling of None values with skip_none=True."""
        transform = transforms.Lambda(lambda x: x * 2)
        wrapper = ListTransformWrapper(transform, skip_none=True)

        tensors = [torch.ones(2, 2), None, torch.ones(3, 3)]
        result = wrapper(tensors)

        assert len(result) == 3
        assert torch.allclose(result[0], torch.ones(2, 2) * 2)
        assert result[1] is None
        assert torch.allclose(result[2], torch.ones(3, 3) * 2)

    def test_none_handling_no_skip(self):
        """Test handling of None values with skip_none=False."""
        transform = transforms.Lambda(lambda x: x * 2)
        wrapper = ListTransformWrapper(transform, skip_none=False)

        tensors = [torch.ones(2, 2), None, torch.ones(3, 3)]

        with pytest.raises(ValueError, match="None value found at index 1"):
            wrapper(tensors)

    def test_invalid_input_type(self):
        """Test error handling for invalid input types."""
        transform = transforms.Lambda(lambda x: x)
        wrapper = ListTransformWrapper(transform)

        with pytest.raises(TypeError, match="Expected list or tuple"):
            wrapper("not a list")

    def test_invalid_tensor_type(self):
        """Test error handling for invalid tensor types in list."""
        transform = transforms.Lambda(lambda x: x)
        wrapper = ListTransformWrapper(transform)

        tensors = [torch.ones(2, 2), "not a tensor", torch.ones(3, 3)]

        with pytest.raises(TypeError, match="Expected torch.Tensor at index 1"):
            wrapper(tensors)

    def test_transform_error_handling(self):
        """Test error handling when transform fails."""

        # Create a transform that will fail
        def failing_transform(x):
            raise ValueError("Transform failed")

        wrapper = ListTransformWrapper(failing_transform)
        tensors = [torch.ones(2, 2)]

        with pytest.raises(RuntimeError, match="Transform failed on tensor at index 0"):
            wrapper(tensors)

    def test_with_torchvision_transforms(self):
        """Test with actual torchvision transforms."""
        # Test with Resize
        resize_transform = transforms.Resize((32, 32))
        wrapper = ListTransformWrapper(resize_transform)

        tensors = [
            torch.rand(3, 64, 64),
            torch.rand(3, 128, 96),
            torch.rand(3, 48, 48),
        ]

        result = wrapper(tensors)

        # All should be resized to 32x32
        for tensor in result:
            assert tensor.shape == (3, 32, 32)

    def test_repr(self):
        """Test string representation."""
        transform = transforms.Lambda(lambda x: x)
        wrapper = ListTransformWrapper(transform, skip_none=False, preserve_type=False)

        repr_str = repr(wrapper)
        assert "ListTransformWrapper" in repr_str
        assert "skip_none=False" in repr_str
        assert "preserve_type=False" in repr_str


class TestBatchListTransformWrapper:
    """Test cases for BatchListTransformWrapper."""

    def test_initialization(self):
        """Test batch wrapper initialization."""
        transform = transforms.Normalize([0.5], [0.5])
        wrapper = BatchListTransformWrapper(transform, batch_size=4, collect_errors=True, device="cpu")

        assert wrapper.transform == transform
        assert wrapper.batch_size == 4
        assert wrapper.collect_errors is True
        assert wrapper.device == torch.device("cpu")

    def test_batch_processing(self):
        """Test batch processing functionality."""
        transform = transforms.Lambda(lambda x: x * 3)
        wrapper = BatchListTransformWrapper(transform, batch_size=2)

        tensors = [
            torch.ones(2, 2),
            torch.ones(3, 3),
            torch.ones(4, 4),
            torch.ones(5, 5),
        ]

        result = wrapper(tensors)

        assert len(result) == len(tensors)
        for i, tensor in enumerate(result):
            assert torch.allclose(tensor, tensors[i] * 3)

    def test_no_batch_size(self):
        """Test processing without batch size (process all at once)."""
        transform = transforms.Lambda(lambda x: x * 2)
        wrapper = BatchListTransformWrapper(transform)  # No batch_size

        tensors = [torch.ones(2, 2), torch.ones(3, 3)]
        result = wrapper(tensors)

        assert len(result) == len(tensors)
        for i, tensor in enumerate(result):
            assert torch.allclose(tensor, tensors[i] * 2)

    def test_error_collection(self):
        """Test error collection mode."""

        def sometimes_failing_transform(x):
            if x.shape[0] == 3:  # Fail on specific shape
                raise ValueError("Intentional failure")
            return x * 2

        wrapper = BatchListTransformWrapper(sometimes_failing_transform, collect_errors=True)

        tensors = [
            torch.ones(2, 2),
            torch.ones(3, 3),  # This will fail
            torch.ones(4, 4),
        ]

        result = wrapper(tensors)

        assert len(result) == len(tensors)
        assert wrapper.has_errors()
        assert len(wrapper.get_errors()) == 1

        # Check that failed tensor is returned unchanged
        assert torch.allclose(result[1], tensors[1])

    def test_device_movement(self):
        """Test device movement functionality."""
        transform = transforms.Lambda(lambda x: x)
        wrapper = BatchListTransformWrapper(transform, device="cpu")

        tensors = [torch.ones(2, 2), torch.ones(3, 3)]
        result = wrapper(tensors)

        # All results should be on CPU
        for tensor in result:
            assert tensor.device.type == "cpu"

    def test_get_errors(self):
        """Test error retrieval methods."""

        def failing_transform(x):
            raise ValueError("Always fails")

        wrapper = BatchListTransformWrapper(failing_transform, collect_errors=True)

        tensors = [torch.ones(2, 2)]
        wrapper(tensors)

        assert wrapper.has_errors()
        errors = wrapper.get_errors()
        assert len(errors) == 1
        assert errors[0][0] == 0  # Index
        assert isinstance(errors[0][1], RuntimeError)  # Error type

    def test_repr(self):
        """Test string representation."""
        transform = transforms.Lambda(lambda x: x)
        wrapper = BatchListTransformWrapper(transform, batch_size=8, collect_errors=True, device="cpu")

        repr_str = repr(wrapper)
        assert "BatchListTransformWrapper" in repr_str
        assert "batch_size=8" in repr_str
        assert "collect_errors=True" in repr_str
        assert "device=cpu" in repr_str


class TestCreateListTransform:
    """Test the factory function."""

    def test_simple_wrapper_creation(self):
        """Test creating simple wrapper."""
        transform = transforms.Lambda(lambda x: x)
        wrapper = create_list_transform(transform)

        assert isinstance(wrapper, ListTransformWrapper)
        assert wrapper.transform == transform

    def test_batch_wrapper_creation(self):
        """Test creating batch wrapper."""
        transform = transforms.Lambda(lambda x: x)
        wrapper = create_list_transform(transform, batch_processing=True, batch_size=4, device="cpu")

        assert isinstance(wrapper, BatchListTransformWrapper)
        assert wrapper.batch_size == 4
        assert wrapper.device == torch.device("cpu")


class TestRealWorldUsage:
    """Test real-world usage scenarios."""

    def test_normalization_different_sizes(self):
        """Test normalization on different sized images."""
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        wrapper = ListTransformWrapper(normalize)

        # Different sized RGB images
        images = [
            torch.rand(3, 224, 224),
            torch.rand(3, 512, 384),
            torch.rand(3, 128, 128),
        ]

        normalized = wrapper(images)

        # Check that normalization was applied (mean should be close to expected)
        for img in normalized:
            assert img.shape[0] == 3  # Still RGB
            # Normalized values should be in a reasonable range
            assert img.min() > -5 and img.max() < 5

    def test_compose_with_resize(self):
        """Test with composed transforms including resize."""
        composed_transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )

        wrapper = ListTransformWrapper(composed_transform)

        # Different sized images
        images = [
            torch.rand(3, 256, 256),
            torch.rand(3, 512, 384),
            torch.rand(3, 128, 96),
        ]

        result = wrapper(images)

        # All should be resized to 224x224 and normalized
        for img in result:
            assert img.shape == (3, 224, 224)
            assert img.min() >= -1 and img.max() <= 1  # Normalized range

    def test_memory_efficient_batch_processing(self):
        """Test memory efficient processing of large lists."""
        # Simple transform
        transform = transforms.Lambda(lambda x: x.clone())

        wrapper = BatchListTransformWrapper(
            transform,
            batch_size=2,
            device="cpu",  # Use CPU to avoid GPU memory issues
        )

        # Create larger list of tensors
        large_tensor_list = [torch.rand(3, 64, 64) for _ in range(10)]

        result = wrapper(large_tensor_list)

        assert len(result) == len(large_tensor_list)
        for i, tensor in enumerate(result):
            assert tensor.shape == large_tensor_list[i].shape


if __name__ == "__main__":
    pytest.main([__file__])
