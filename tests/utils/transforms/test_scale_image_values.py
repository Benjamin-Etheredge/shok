from shok.utils.transforms.scale_image_values import ScaleImageValues


def test_init_default_values():
    scaler = ScaleImageValues()
    assert scaler.min == 0
    assert scaler.max == 255


def test_init_custom_values():
    scaler = ScaleImageValues(min=10, max=100)
    assert scaler.min == 10
    assert scaler.max == 100
