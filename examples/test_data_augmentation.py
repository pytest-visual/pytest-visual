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


def test_show_augmentations(visual: VisualFixture, fix_seeds: None):
    base_image = torchvision.io.read_image("examples/assets/english_springer.jpg")
    augmented = [standardize(augment(base_image).numpy()) for _ in range(6)]
    labels = [f"Doge {i}" for i in range(1, 7)]

    visual.print("Augmented images")
    visual.show_images(augmented, labels=labels)
