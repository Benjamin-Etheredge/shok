from typing import Any, cast

import torch

from shok.utils import functions


class PassRound(torch.nn.Module):
    """
    A custom torch.nn.Module that applies a soft rounding operation to the input tensor or list of tensors.

    Args:
        x (Union[torch.Tensor, List[torch.Tensor]]): The input tensor(s) to be rounded.
        y (optional): An optional secondary input, passed through unchanged.

    Returns:
        Tuple[Union[torch.Tensor, List[torch.Tensor]], Any]: A tuple containing the rounded tensor(s) and the optional secondary input.

    Note:
        The actual rounding logic is implemented in `functions.PassRound.apply`.

    """

    def forward(
        self, x: torch.Tensor | (list[torch.Tensor] | tuple[torch.Tensor]), y=None
    ) -> tuple[torch.Tensor | list[torch.Tensor], Any]:
        """
        Applies a placeholder soft rounding operation to the input tensor(s).

        Args:
            x (Union[torch.Tensor, List[torch.Tensor]]): Input tensor(s) to be processed.
            y (optional): Additional input, currently unused.

        Returns:
            Tuple[Union[torch.Tensor, List[torch.Tensor]], Any]: A tuple containing the processed tensor(s) and the second input (y).

        """
        # Handle both single tensor and list of tensors
        if isinstance(x, list):
            # Apply PassRound to each tensor in the list
            rounded_list: list[torch.Tensor] = [cast(torch.Tensor, functions.PassRound.apply(tensor)) for tensor in x]
            return rounded_list, y
        else:
            # Apply PassRound to the single tensor
            rounded_tensor: torch.Tensor = cast(torch.Tensor, functions.PassRound.apply(x))
            return rounded_tensor, y
