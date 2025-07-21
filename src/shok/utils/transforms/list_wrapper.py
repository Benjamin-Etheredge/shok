"""
Transform wrapper for applying transforms to lists of tensors with different sizes.

This module provides a wrapper that can apply any torchvision transform to a list of tensors
individually, handling tensors of different sizes gracefully.
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class ListTransformWrapper(nn.Module):
    """
    A PyTorch Module wrapper that applies a torchvision transform to each element in a list of tensors individually.

    This is useful when you have a list of tensors with different sizes that need to be transformed
    using the same transform. The wrapper applies the transform to each tensor individually and
    returns a list of transformed tensors.

    Inherits from nn.Module to be compatible with PyTorch's module system, allowing it to be part
    of larger neural network architectures and benefit from features like device movement,
    state_dict serialization, and train/eval modes.

    Args:
        transform: The torchvision transform or callable to apply to each tensor.
        skip_none: Whether to skip None values in the input list. Default: True.
        preserve_type: Whether to preserve the original container type (list/tuple). Default: True.

    Example:
        ```python
        from torchvision import transforms
        from shok.utils.transforms.list_wrapper import ListTransformWrapper

        # Create a transform
        base_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
            transforms.Resize((224, 224))
        ])

        # Wrap it for list processing
        list_transform = ListTransformWrapper(base_transform)

        # Apply to list of different sized tensors
        tensors = [
            torch.rand(3, 256, 256),  # Different sizes
            torch.rand(3, 512, 384),
            torch.rand(3, 128, 128),
        ]

        transformed = list_transform(tensors)
        # Each tensor is now transformed individually

        # Can be used as part of nn.Sequential or other modules
        model = nn.Sequential(
            list_transform,
            SomeOtherModule()
        )
        ```

    Note:
        - Each tensor in the input list is transformed independently
        - The transform is applied to each tensor using the same parameters
        - Output maintains the same list structure as input
        - Supports any callable transform (torchvision transforms, custom functions, etc.)
        - Inherits all nn.Module benefits (device movement, serialization, etc.)

    """

    def __init__(self, transform: Callable[[Tensor], Tensor], skip_none: bool = True, preserve_type: bool = True):
        super().__init__()
        self.transform = transform
        self.skip_none = skip_none
        self.preserve_type = preserve_type

        # Register transform as a module if it's a nn.Module
        if isinstance(transform, nn.Module):
            self.add_module("transform", transform)

    def forward(self, tensor_list: list[Tensor] | tuple[Tensor, ...]) -> list[Tensor] | tuple[Tensor, ...]:
        """
        Forward pass - apply the transform to each tensor in the input list.

        Args:
            tensor_list: List or tuple of tensors to transform. Can contain None values if skip_none=True.

        Returns:
            List or tuple of transformed tensors (same type as input if preserve_type=True).

        Raises:
            TypeError: If input is not a list or tuple.
            RuntimeError: If transform fails on any tensor.

        """
        return self._apply_transforms(tensor_list)

    def __call__(self, tensor_list: list[Tensor] | tuple[Tensor, ...]) -> list[Tensor] | tuple[Tensor, ...]:
        """Alias for forward() to maintain backward compatibility."""
        return self.forward(tensor_list)

    def _apply_transforms(self, tensor_list: list[Tensor] | tuple[Tensor, ...]) -> list[Tensor] | tuple[Tensor, ...]:
        """Internal method to apply transforms with proper error handling."""
        if not isinstance(tensor_list, list | tuple):
            raise TypeError(f"Expected list or tuple, got {type(tensor_list)}")

        original_type = type(tensor_list)
        transformed_tensors = []

        for i, tensor in enumerate(tensor_list):
            if tensor is None and self.skip_none:
                transformed_tensors.append(None)
                continue

            if tensor is None and not self.skip_none:
                raise ValueError(f"None value found at index {i} but skip_none=False")

            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor at index {i}, got {type(tensor)}")

            try:
                # Ensure tensor is on the same device as the module if transform is a module
                if isinstance(self.transform, nn.Module) and next(self.transform.parameters(), None) is not None:
                    device = next(self.transform.parameters()).device
                    tensor = tensor.to(device)

                transformed_tensor = self.transform(tensor)
                transformed_tensors.append(transformed_tensor)
            except Exception as e:
                raise RuntimeError(f"Transform failed on tensor at index {i}: {e!s}") from e

        # Preserve original container type if requested
        if self.preserve_type and isinstance(original_type, tuple):
            return tuple(transformed_tensors)
        else:
            return transformed_tensors

    def extra_repr(self) -> str:
        """Extra representation for better debugging."""
        return f"skip_none={self.skip_none}, preserve_type={self.preserve_type}"


class BatchListTransformWrapper(nn.Module):
    """
    A more advanced PyTorch Module wrapper that can handle batched operations and provides additional utilities.

    This wrapper extends the basic ListTransformWrapper with features like:
    - Batch processing for efficiency
    - Progress tracking for large lists
    - Error collection mode
    - Memory management options
    - Full PyTorch Module compatibility

    Args:
        transform: The torchvision transform or callable to apply.
        batch_size: Process tensors in batches of this size. If None, process all at once. Default: None.
        skip_none: Whether to skip None values. Default: True.
        preserve_type: Whether to preserve container type. Default: True.
        collect_errors: Whether to collect errors instead of raising immediately. Default: False.
        device: Device to move tensors to before processing. Default: None.

    Example:
        ```python
        # For large lists of tensors
        batch_transform = BatchListTransformWrapper(
            transform=transforms.Resize((224, 224)),
            batch_size=32,  # Process 32 at a time
            device='cuda'   # Move to GPU for processing
        )

        # Process large list efficiently
        large_tensor_list = [torch.rand(3, h, w) for h, w in zip(heights, widths)]
        result = batch_transform(large_tensor_list)

        # Can be moved to different devices
        batch_transform = batch_transform.cuda()

        # Can be serialized/deserialized
        torch.save(batch_transform.state_dict(), 'batch_transform.pth')
        ```

    """

    def __init__(
        self,
        transform: Callable[[Tensor], Tensor],
        batch_size: int | None = None,
        skip_none: bool = True,
        preserve_type: bool = True,
        collect_errors: bool = False,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        self.transform = transform
        self.batch_size = batch_size
        self.skip_none = skip_none
        self.preserve_type = preserve_type
        self.collect_errors = collect_errors
        self._target_device = device  # Store as private to avoid conflicts
        self.errors = []

        # Register transform as a module if it's a nn.Module
        if isinstance(transform, nn.Module):
            self.add_module("transform", transform)

        # Register device as a buffer if specified
        if device is not None:
            self.register_buffer("_device_tensor", torch.tensor(0, device=device))

    @property
    def target_device(self) -> torch.device | None:
        """Get the target device for tensor processing."""
        if hasattr(self, "_device_tensor"):
            return self._device_tensor.device
        if isinstance(self._target_device, str):
            return torch.device(self._target_device)
        return self._target_device

    @property
    def device(self) -> torch.device | None:
        """Get the target device - alias for target_device for backward compatibility."""
        return self.target_device

    def forward(self, tensor_list: list[Tensor] | tuple[Tensor, ...]) -> list[Tensor] | tuple[Tensor, ...]:
        """
        Forward pass - apply transform to tensor list with batch processing.

        Args:
            tensor_list: List or tuple of tensors to transform.

        Returns:
            List or tuple of transformed tensors.

        """
        return self._apply_batch_transforms(tensor_list)

    def __call__(self, tensor_list: list[Tensor] | tuple[Tensor, ...]) -> list[Tensor] | tuple[Tensor, ...]:
        """Alias for forward() to maintain backward compatibility."""
        return self.forward(tensor_list)

    def _apply_batch_transforms(
        self, tensor_list: list[Tensor] | tuple[Tensor, ...]
    ) -> list[Tensor] | tuple[Tensor, ...]:
        """Internal method to apply batch transforms."""
        if not isinstance(tensor_list, list | tuple):
            raise TypeError(f"Expected list or tuple, got {type(tensor_list)}")

        original_type = type(tensor_list)
        self.errors.clear()

        # Process in batches if specified
        if self.batch_size is not None:
            return self._process_in_batches(tensor_list, original_type)
        else:
            return self._process_all(tensor_list, original_type)

    def _process_all(self, tensor_list: list[Tensor] | tuple, original_type) -> list[Tensor] | tuple:
        """Process all tensors at once."""
        transformed_tensors = []

        for i, tensor in enumerate(tensor_list):
            transformed = self._process_single_tensor(tensor, i)
            transformed_tensors.append(transformed)

        return self._return_result(transformed_tensors, original_type)

    def _process_in_batches(self, tensor_list: list[Tensor] | tuple, original_type) -> list[Tensor] | tuple:
        """Process tensors in batches."""
        transformed_tensors = []

        # Ensure batch_size is not None for batch processing
        batch_size = self.batch_size if self.batch_size is not None else len(tensor_list)

        for i in range(0, len(tensor_list), batch_size):
            batch = tensor_list[i : i + batch_size]
            batch_results = []

            for j, tensor in enumerate(batch):
                transformed = self._process_single_tensor(tensor, i + j)
                batch_results.append(transformed)

            transformed_tensors.extend(batch_results)

            # Optional: Clear GPU cache between batches
            if self.target_device and "cuda" in str(self.target_device):
                torch.cuda.empty_cache()

        return self._return_result(transformed_tensors, original_type)

    def _process_single_tensor(self, tensor: Any, index: int) -> Any:
        """Process a single tensor with error handling."""
        if tensor is None and self.skip_none:
            return None

        if tensor is None and not self.skip_none:
            error_msg = f"None value found at index {index} but skip_none=False"
            if self.collect_errors:
                self.errors.append((index, ValueError(error_msg)))
                return None
            else:
                raise ValueError(error_msg)

        if not isinstance(tensor, torch.Tensor):
            error_msg = f"Expected torch.Tensor at index {index}, got {type(tensor)}"
            if self.collect_errors:
                self.errors.append((index, TypeError(error_msg)))
                return tensor  # Return original if collecting errors
            else:
                raise TypeError(error_msg)

        try:
            # Move to device if specified
            if self.target_device is not None:
                tensor = tensor.to(self.target_device)

            transformed_tensor = self.transform(tensor)
            return transformed_tensor

        except Exception as e:
            error_msg = f"Transform failed on tensor at index {index}: {e!s}"
            if self.collect_errors:
                self.errors.append((index, RuntimeError(error_msg)))
                return tensor  # Return original if collecting errors
            else:
                raise RuntimeError(error_msg) from e

    def _return_result(self, transformed_tensors: list, original_type) -> list[Tensor] | tuple:
        """Return result in appropriate format."""
        # If there were errors and we're collecting them, optionally raise
        if self.errors and not self.collect_errors:
            # This shouldn't happen, but just in case
            raise RuntimeError(f"Collected {len(self.errors)} errors during processing")

        # Preserve original container type if requested
        if self.preserve_type and isinstance(original_type, tuple):
            return tuple(transformed_tensors)
        else:
            return transformed_tensors

    def get_errors(self) -> list[tuple]:
        """Get list of (index, error) tuples from last processing."""
        return self.errors.copy()

    def has_errors(self) -> bool:
        """Check if there were any errors in the last processing."""
        return len(self.errors) > 0

    def __repr__(self) -> str:
        """Return a string representation of the wrapper."""
        return (
            f"{self.__class__.__name__}("
            f"transform={self.transform}, "
            f"batch_size={self.batch_size}, "
            f"skip_none={self.skip_none}, "
            f"preserve_type={self.preserve_type}, "
            f"collect_errors={self.collect_errors}, "
            f"device={self.device})"
        )


def create_list_transform(
    transform: Callable[[Tensor], Tensor], batch_processing: bool = False, **kwargs
) -> ListTransformWrapper | BatchListTransformWrapper:
    """
    Factory function to create appropriate list transform wrapper.

    Args:
        transform: The transform to wrap.
        batch_processing: Whether to use batch processing capabilities.
        **kwargs: Additional arguments passed to the wrapper constructor.

    Returns:
        ListTransformWrapper or BatchListTransformWrapper instance.

    Example:
        ```python
        # Simple wrapper
        simple_wrapper = create_list_transform(transforms.Resize((224, 224)))

        # Batch processing wrapper
        batch_wrapper = create_list_transform(
            transforms.Normalize([0.5], [0.5]),
            batch_processing=True,
            batch_size=16,
            device='cuda'
        )
        ```

    """
    if batch_processing:
        return BatchListTransformWrapper(transform, **kwargs)
    else:
        return ListTransformWrapper(transform, **kwargs)
