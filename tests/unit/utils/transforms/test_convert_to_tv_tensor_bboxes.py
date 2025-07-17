import torch
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

from shok.utils.transforms.convert_to_tv_tensor_bboxes import ConvertToTVTensorBBoxes


def test_forward_with_boxes():
    x = torch.randn(3, 224, 224)  # Example image tensor
    boxes = torch.tensor([[10, 20, 50, 60], [30, 40, 70, 80]], dtype=torch.float32)
    y = {"boxes": boxes.clone()}
    module = ConvertToTVTensorBBoxes()
    x_out, y_out = module.forward(x, y)

    assert torch.equal(x_out, x)
    assert isinstance(y_out["boxes"], BoundingBoxes)
    assert y_out["boxes"].format == BoundingBoxFormat.XYXY
    assert y_out["boxes"].canvas_size == (224, 224)
    assert y_out["boxes"].dtype == torch.float32
    assert torch.allclose(y_out["boxes"], boxes)


def test_forward_without_boxes_key():
    x = torch.randn(3, 100, 100)
    y = {"labels": torch.tensor([1, 2])}
    module = ConvertToTVTensorBBoxes()
    x_out, y_out = module.forward(x, y)
    assert torch.equal(x_out, x)
    assert "boxes" not in y_out


def test_forward_with_none_y():
    x = torch.randn(3, 50, 50)
    module = ConvertToTVTensorBBoxes()
    x_out, y_out = module.forward(x, None)
    assert torch.equal(x_out, x)
    assert y_out is None


def test_forward_with_empty_boxes():
    x = torch.randn(3, 64, 64)
    boxes = torch.empty((0, 4), dtype=torch.float32)
    y = {"boxes": boxes.clone()}
    module = ConvertToTVTensorBBoxes()
    x_out, y_out = module.forward(x, y)
    assert torch.equal(x_out, x)
    assert isinstance(y_out["boxes"], BoundingBoxes)
    assert y_out["boxes"].shape[0] == 0
