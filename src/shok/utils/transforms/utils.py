from torchvision.transforms import v2


def default_patched_image_mutator():
    """Default image mutator for patching images."""
    return v2.Compose(
        [
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=(0, 360), expand=False, center=None, fill=None),
            v2.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # Translate by 10%
                scale=(0.9, 1.1),  # Scale by 10%
                shear=None,
            ),
        ]
    )
