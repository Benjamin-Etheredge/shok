"""
ListCompose: A torchvision.v2.Compose-like class that handles both single tensors and lists of tensors.

This module provides a drop-in replacement for torchvision.v2.Compose that can natively
handle both single tensors and lists of tensors without requiring additional wrappers.
"""

from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast

import torch


class ListCompose(torch.nn.Module):
    """
    Composes several transforms together, similar to torchvision.v2.Compose, but with native support for both single tensors and lists of tensors.

    This class can be used as a drop-in replacement for torchvision.v2.Compose when you need
    to apply the same sequence of transforms to either single tensors or lists of tensors
    of potentially different sizes.

    Args:
        transforms (Sequence[Callable]): List of transforms to compose. Each transform should
            accept (x, y) parameters where x can be either torch.Tensor or List[torch.Tensor]
            and y is optional additional data.

    Example:
        >>> from shok.utils.transforms import ListCompose, PassRound, ScaleImageValues
        >>> from torchvision.transforms import v2
        >>>
        >>> # Create a composition of transforms
        >>> compose = ListCompose([
        ...     PassRound(),
        ...     ScaleImageValues(min=0, max=255),
        ...     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ... ])
        >>>
        >>> # Use with single tensor
        >>> single_tensor = torch.rand(3, 224, 224)
        >>> result, y = compose(single_tensor)
        >>>
        >>> # Use with list of tensors
        >>> tensor_list = [torch.rand(3, 224, 224), torch.rand(3, 256, 256)]
        >>> result_list, y = compose(tensor_list)

    Note:
        - All transforms in the composition should support the (x, y) signature
        - When processing lists, each tensor is processed individually through the entire pipeline
        - The y parameter is passed through all transforms unchanged
        - Input structure is preserved: tensor input → tensor output, list input → list output

    """

    def __init__(self, transforms: Sequence[Any]):
        """
        Initialize the ListCompose with a sequence of transforms.

        Args:
            transforms (Sequence[Any]): Sequence of callable transforms to apply in order.
                Each transform should accept (x, y=None) parameters and return (x, y) tuple.

        Raises:
            TypeError: If transforms is not iterable or contains non-callable items.

        """
        super().__init__()
        if not isinstance(transforms, Iterable):
            raise TypeError("transforms should be an iterable of callables")

        self.transforms = torch.nn.ModuleList()
        for transform in transforms:
            if callable(transform):
                # If it's already a nn.Module, add it directly
                if isinstance(transform, torch.nn.Module):
                    self.transforms.append(transform)
                else:
                    # Wrap non-module callables in a simple wrapper
                    self.transforms.append(_CallableWrapper(transform))
            else:
                raise TypeError(f"Transform {transform} is not callable")

    def forward(
        self, x: torch.Tensor | list[torch.Tensor], y: Any = None
    ) -> tuple[torch.Tensor | list[torch.Tensor], Any]:
        """
        Apply all transforms in sequence to the input.

        Args:
            x (Union[torch.Tensor, List[torch.Tensor]]): Input tensor(s) to be transformed.
            y (Any, optional): Additional data to pass through transforms. Defaults to None.

        Returns:
            Tuple[Union[torch.Tensor, List[torch.Tensor]], Any]: Tuple containing:
                - Transformed tensor(s) with same structure as input
                - The y parameter (potentially modified by transforms)

        """
        # Handle both single tensor and list of tensors
        if isinstance(x, list):
            return self._process_tensor_list(x, y)
        else:
            return self._process_single_tensor(x, y)

    def _process_single_tensor(self, x: torch.Tensor, y: Any) -> tuple[torch.Tensor, Any]:
        """
        Process a single tensor through all transforms.

        Args:
            x (torch.Tensor): Input tensor.
            y (Any): Additional data.

        Returns:
            Tuple[torch.Tensor, Any]: Transformed tensor and additional data.

        """
        for transform in self.transforms:
            x, y = transform(x, y)
        return x, y

    def _process_tensor_list(self, tensor_list: list[torch.Tensor], y: Any) -> tuple[list[torch.Tensor], Any]:
        """
        Process a list of tensors through all transforms.

        Each tensor in the list is processed individually through the entire transform pipeline.

        Args:
            tensor_list (List[torch.Tensor]): List of input tensors.
            y (Any): Additional data.

        Returns:
            Tuple[List[torch.Tensor], Any]: List of transformed tensors and additional data.

        """
        result_list = []
        for tensor in tensor_list:
            # Process each tensor individually through all transforms
            transformed_tensor, y = self._process_single_tensor(tensor, y)
            result_list.append(transformed_tensor)

        return result_list, y

    def __len__(self) -> int:
        """Return the number of transforms in the composition."""
        return len(self.transforms)

    def __getitem__(self, index: int) -> torch.nn.Module:
        """Get a transform by index."""
        return self.transforms[index]

    def __repr__(self) -> str:
        """Return a string representation of the composition."""
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string


class _CallableWrapper(torch.nn.Module):
    """
    Wrapper to make non-Module callables compatible with nn.ModuleList.

    This is used internally to wrap transform functions that aren't nn.Module instances.
    """

    def __init__(self, transform: Callable):
        super().__init__()
        self.transform = transform

    def forward(
        self, x: torch.Tensor | list[torch.Tensor], y: Any = None
    ) -> tuple[torch.Tensor | list[torch.Tensor], Any]:
        """Apply the wrapped transform."""
        # Try to call with (x, y) signature first
        try:
            result = self.transform(x, y)
            if isinstance(result, tuple) and len(result) == 2:
                # Cast to satisfy type checker
                return cast(tuple[torch.Tensor | list[torch.Tensor], Any], result)
            else:
                # If transform doesn't return tuple, assume it only transformed x
                transformed_x = cast(torch.Tensor | list[torch.Tensor], result)
                return transformed_x, y
        except TypeError:
            # Fallback to single argument signature
            try:
                transformed_x = self.transform(x)
                return cast(torch.Tensor | list[torch.Tensor], transformed_x), y
            except TypeError as e:
                raise TypeError(
                    f"Transform {self.transform} doesn't accept expected signature (x, y) or (x): {e}"
                ) from e

    def __repr__(self) -> str:
        return f"_CallableWrapper({self.transform})"
