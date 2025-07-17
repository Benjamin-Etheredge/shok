import torch


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
