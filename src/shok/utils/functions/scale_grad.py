import torch


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
