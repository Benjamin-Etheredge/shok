import torch

import shok.utils.functions.soft_round


class SoftRound(torch.nn.Module):
    """
    Transform to use the soft round function for adversarial training.

    This is something being explored. Since rounding is not differentiable,
    additional logic is needed to ensure gradients can flow through the operation.

    This way does the rounding, but then calculates what the multiplier factor was.
    Then this value is used to scale the gradient.
    """

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        """
        Applies soft rounding to the input tensor using the SoftRound function.

        Args:
            x (torch.Tensor): Input tensor to be processed.
            y (optional): An optional secondary input, not used in the transformation.

        Returns:
            Tuple[torch.Tensor, Any]: A tuple containing the transformed tensor and the optional secondary input.

        """
        # Placeholder for soft rounding logic
        return shok.utils.functions.soft_round.SoftRound.apply(x), y
