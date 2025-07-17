import torch


class WholePixelOptimizer(torch.optim.Optimizer):
    """
    A custom optimizer that operates on the pixel level by stepping by whole pixel values.

    This approach is common in adversarial patch training.

    """

    def __init__(self, params, lr=0.01):
        """
        Initializes the optimizer with the given parameters and learning rate.

        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float, optional): Learning rate. Default is 0.01.

        """
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        This method updates the parameters by adding the sign of their gradients multiplied by a rounded learning rate.
        The learning rate is rounded to the nearest integer and is at least 1.0.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            loss: The loss value returned by the closure, if provided; otherwise, None.

        """
        loss = None
        if closure is not None:
            loss = closure()

        rounded_lr = max(round(self.param_groups[0]["lr"], 0), 1.0)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                p.data.add_(torch.sign(p.grad) * rounded_lr)

        return loss


class SubPixelOptimizer(torch.optim.Optimizer):
    """A custom optimizer that operates on the pixel level."""

    def __init__(self, params, lr=20.0):
        """
        Initializes the optimizer with the given parameters and learning rate.

        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float, optional): Learning rate. Defaults to 20.0.

        """
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        If a closure is provided, it is called and its result is returned as the loss.

        For each parameter group and each parameter within the group:
            - If the parameter has a gradient, normalize the gradient by its maximum absolute value (if non-zero).
            - Update the parameter data by adding the scaled gradient multiplied by the learning rate.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            The loss value returned by the closure, if provided; otherwise, None.

        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                max_abs = p.grad.abs().max()
                p.grad = p.grad / max_abs if max_abs > 0 else p.grad

                p.data.add_(p.grad * group["lr"])

        return loss
