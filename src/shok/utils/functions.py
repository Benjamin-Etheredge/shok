import torch


class SoftRound(torch.autograd.Function):
    """
    A custom autograd function that performs a "soft" rounding operation on a tensor.

    This function rounds the input tensor to the nearest integer in the forward pass.
    During the backward pass, it computes gradients using a ratio between the input and the rounded value,
    which helps to maintain meaningful gradients and avoid division by zero.

    Forward Pass:
        - Rounds the input tensor using torch.round.
        - Saves a delta ratio for gradient computation if gradients are required.

    Backward Pass:
        - Uses the saved delta ratio to scale the incoming gradient, allowing gradients to flow through the rounding operation.

    Note:
        - The gradient computation is an approximation and differs from the original paper, but aims to provide stable gradients.

        x (torch.Tensor): Input tensor to be rounded.

        torch.Tensor: Rounded tensor (forward), gradient tensor (backward).

    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the soft rounding operation.

        Performs a forward pass that rounds the input tensor `x` element-wise without tracking gradients.
        If gradients are required, computes a delta ratio to approximate gradient flow for non-differentiable rounding.

        Args:
            ctx: Context object used to save information for backward computation.
            x (torch.Tensor): Input tensor to be rounded.

        Returns:
            torch.Tensor: Tensor with each element rounded to the nearest integer.

        Notes:
            - If `x.requires_grad` is True, saves a delta ratio for use in the backward pass.
            - The delta ratio is calculated as `x / rounded` where `rounded != 0`, otherwise set to zero.
            - If both `x` and `rounded` are zero, the delta ratio is set to one.
            - This gradient approximation is not identical to the original paper, but serves as a practical alternative.

        """
        with torch.no_grad():
            rounded = torch.round(x)

        if x.requires_grad:
            # Calculate the ratio of the delta to the rounded value
            # This is to ensure that the gradient is not too large
            # and to prevent division by zero
            # NOTE this is not the same as the original paper but it is a potential good approximation
            delta_ratio = torch.where(rounded != 0, x / rounded, torch.zeros_like(x))
            delta_ratio = torch.where(torch.logical_and(x == 0, rounded == 0), torch.ones_like(x), delta_ratio)

            ctx.save_for_backward(delta_ratio)

        return rounded

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Computes the gradient of the output with respect to the input using the saved delta ratio.

        Args:
            ctx: The context object containing saved tensors from the forward pass.
            grad_output (torch.Tensor): The gradient of the loss with respect to the output.

        Returns:
            torch.Tensor: The gradient of the loss with respect to the input.

        """
        delta_ratio, *_ = ctx.saved_tensors
        delta = grad_output * delta_ratio
        return delta


class PassRound(torch.autograd.Function):
    """
    Simple rounding function that passes the gradient unchanged.

    A custom autograd function that rounds each element of the input tensor to the nearest integer
    during the forward pass, while passing the gradient unchanged during the backward pass.

    Class Methods:
        forward(ctx, x):

        backward(ctx, grad_output):
            Passes the gradient of the loss with respect to the output unchanged.

    """

    @staticmethod
    def forward(ctx, x):
        """
        Rounds each element of the input tensor to the nearest integer.

        Args:
            ctx: Context object (not used in this function).
            x (torch.Tensor): Input tensor to be rounded.

        Returns:
            torch.Tensor: Tensor with each element rounded to the nearest integer.

        """
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the gradient of the output with respect to the input during the backward pass.

        Args:
            ctx: Context object containing information from the forward pass (not used here).
            grad_output: The gradient of the loss with respect to the output.

        Returns:
            The gradient of the loss with respect to the input, unchanged.

        """
        return grad_output


class ScaleGrad(torch.autograd.Function):
    """
    A custom autograd function that scales gradients during the backward pass.

    The forward pass returns the input unchanged. During the backward pass, the gradient is normalized
    by dividing it by its maximum absolute value, unless the maximum is zero, in which case the gradient
    is returned unchanged.

    Methods:
        forward(ctx, input):
            Returns the input tensor as-is.

        backward(ctx, grad_output):
            Normalizes the gradient by dividing by its maximum absolute value if greater than zero;
            otherwise, returns the gradient unchanged.

    Usage:
        Use ScaleGrad.apply(input) to apply the custom gradient scaling in your computation graph.

    """

    @staticmethod
    def forward(ctx, input):
        """
        Performs a forward pass by returning the input as-is.

        Args:
            ctx: Context object (typically used in autograd functions, but unused here).
            input: The input data to be returned.

        Returns:
            The input data unchanged.

        """
        return input

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the normalized gradient for the backward pass.

        Args:
            ctx: Context object containing information for the backward computation (unused).
            grad_output (Tensor): The gradient output tensor from the subsequent layer.

        Returns:
            Tensor: The normalized gradient tensor, divided by the maximum absolute value of
            grad_output if it is greater than zero; otherwise, returns grad_output unchanged.

        """
        grad_max = grad_output.abs().max()
        return grad_output / grad_max if grad_max > 0 else grad_output
