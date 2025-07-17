import pytest

from shok.utils.transforms.scale_apply_patch import ScaleApplyPatch


def test_default_init():
    sap = ScaleApplyPatch()
    assert sap.scale == 0.25
    assert sap.preserve_aspect_ratio is True


@pytest.mark.parametrize(
    "scale, preserve_aspect_ratio",
    [
        (0.5, True),
        (1.0, False),
        (0.1, True),
        (0.75, False),
    ],
)
def test_custom_init(scale, preserve_aspect_ratio):
    sap = ScaleApplyPatch(scale=scale, preserve_aspect_ratio=preserve_aspect_ratio)
    assert sap.scale == scale
    assert sap.preserve_aspect_ratio == preserve_aspect_ratio
