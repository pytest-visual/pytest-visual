import torch
import torchvision
import torchvision.transforms as transforms

from visual.interface import VisualFixture, standardize


def augment(image: torch.Tensor) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),  # type: ignore
            transforms.Pad(20, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.CenterCrop((224, 224)),
        ]
    )
    image = transform(image)

    return image


# fmt: off

# `visual` fixture indicates that the test results must be manually verified
# `fix_seeds` fixture ensures deterministic results
def test_show_augmentations(visual: VisualFixture, fix_seeds):
    # Create augmented images
    base_image = torchvision.io.read_image("examples/assets/doge.jpg")  # Load the base image
    augmented = [augment(base_image) for _ in range(6)]  # Your augmentation function
    augmented = [image.numpy() for image in augmented]  # pytest-visual accepts only numpy images
    augmented = [standardize(image) for image in augmented]  # Standardize image to uint8 with [0, 255] range and HWC format

    # Show helpful text with images
    visual.text("Doggos augmented")
    visual.images(augmented)

# fmt: on
