import pytest
import torch

from shok.utils.transforms.target_insurance import TargetInsurance


@pytest.fixture
def input_tensor():
    return torch.randn(3, 224, 224)


def test_forward_with_missing_boxes_and_labels(input_tensor):
    transform = TargetInsurance()
    y = {}
    x_out, y_out = transform.forward(input_tensor, y)
    assert torch.equal(x_out, input_tensor)
    assert "boxes" in y_out
    assert "labels" in y_out
    assert y_out["boxes"].shape == (0, 4)
    assert y_out["boxes"].dtype == torch.float32
    assert y_out["labels"].shape == (0,)
    assert y_out["labels"].dtype == torch.int64


def test_forward_with_existing_boxes_and_labels(input_tensor):
    transform = TargetInsurance()
    boxes = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    labels = torch.tensor([1], dtype=torch.int64)
    y = {"boxes": boxes, "labels": labels}
    x_out, y_out = transform.forward(input_tensor, y)
    assert torch.equal(x_out, input_tensor)
    assert torch.equal(y_out["boxes"], boxes)
    assert torch.equal(y_out["labels"], labels)


def test_forward_with_only_boxes(input_tensor):
    transform = TargetInsurance()
    boxes = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    y = {"boxes": boxes}
    x_out, y_out = transform.forward(input_tensor, y)
    assert torch.equal(x_out, input_tensor)
    assert torch.equal(y_out["boxes"], boxes)
    assert "labels" in y_out
    assert y_out["labels"].shape == (0,)
    assert y_out["labels"].dtype == torch.int64


def test_forward_with_only_labels(input_tensor):
    transform = TargetInsurance()
    labels = torch.tensor([1], dtype=torch.int64)
    y = {"labels": labels}
    x_out, y_out = transform.forward(input_tensor, y)
    assert torch.equal(x_out, input_tensor)
    assert torch.equal(y_out["labels"], labels)
    assert "boxes" in y_out
    assert y_out["boxes"].shape == (0, 4)
    assert y_out["boxes"].dtype == torch.float32
