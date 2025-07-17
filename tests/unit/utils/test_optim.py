import torch

from shok.utils.optim import SubPixelOptimizer, WholePixelOptimizer


def test_step_updates_parameters_by_whole_pixel():
    # Create a parameter tensor with requires_grad=True
    param = torch.nn.Parameter(torch.tensor([0.0, 1.0, -2.0]))
    # Assign gradients
    param.grad = torch.tensor([2.5, -3.0, 0.0])
    # Use a non-integer learning rate to test rounding
    optimizer = WholePixelOptimizer([param], lr=0.7)
    # Step
    optimizer.step()
    # lr rounded to 1.0, sign([2.5, -3.0, 0.0]) = [1, -1, 0]
    expected = torch.tensor([1.0, 0.0, -2.0])
    assert torch.allclose(param.data, expected)


def test_step_with_higher_lr():
    param = torch.nn.Parameter(torch.tensor([5.0, -5.0]))
    param.grad = torch.tensor([-1.0, 1.0])
    optimizer = WholePixelOptimizer([param], lr=2.3)
    optimizer.step()
    # lr rounded to 2.0, sign([-1, 1]) = [-1, 1]
    expected = torch.tensor([3.0, -3.0])
    assert torch.allclose(param.data, expected)


def test_step_with_zero_grad_does_not_update():
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    param.grad = torch.tensor([0.0, 0.0])
    optimizer = WholePixelOptimizer([param], lr=1.5)
    optimizer.step()
    # No update since sign(0) = 0
    expected = torch.tensor([1.0, 2.0])
    assert torch.allclose(param.data, expected)


def test_step_with_no_grad_skips_param():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    param.grad = None
    optimizer = WholePixelOptimizer([param], lr=1.0)
    optimizer.step()
    # No update since grad is None
    assert torch.allclose(param.data, torch.tensor([1.0]))


def test_step_returns_loss_from_closure():
    param = torch.nn.Parameter(torch.tensor([0.0]))
    param.grad = torch.tensor([1.0])
    optimizer = WholePixelOptimizer([param], lr=1.0)

    def closure():
        return 42

    loss = optimizer.step(closure)
    assert loss == 42


def test_subpixel_step_normalizes_and_updates():
    param = torch.nn.Parameter(torch.tensor([2.0, -4.0, 6.0]))
    param.grad = torch.tensor([2.0, -4.0, 8.0])
    optimizer = SubPixelOptimizer([param], lr=10.0)
    optimizer.step()
    # max_abs = 8.0, grad = [0.25, -0.5, 1.0]
    expected = torch.tensor([2.0 + 2.5, -4.0 - 5.0, 6.0 + 10.0])
    assert torch.allclose(param.data, expected)


def test_subpixel_step_with_zero_max_abs_grad():
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    param.grad = torch.tensor([0.0, 0.0])
    optimizer = SubPixelOptimizer([param], lr=5.0)
    optimizer.step()
    # grad remains zero, no update
    expected = torch.tensor([1.0, 2.0])
    assert torch.allclose(param.data, expected)


def test_subpixel_step_with_no_grad_skips_param():
    param = torch.nn.Parameter(torch.tensor([3.0]))
    param.grad = None
    optimizer = SubPixelOptimizer([param], lr=7.0)
    optimizer.step()
    # No update since grad is None
    assert torch.allclose(param.data, torch.tensor([3.0]))


def test_subpixel_step_returns_loss_from_closure():
    param = torch.nn.Parameter(torch.tensor([0.0]))
    param.grad = torch.tensor([1.0])
    optimizer = SubPixelOptimizer([param], lr=1.0)

    def closure():
        return 123

    loss = optimizer.step(closure)
    assert loss == 123
