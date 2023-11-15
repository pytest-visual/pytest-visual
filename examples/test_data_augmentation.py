
import plotly.express as px
import torch
import torchvision
import torchvision.transforms as transforms
from plotly.subplots import make_subplots

from visual.interface import VisualFixture


def augment(image: torch.Tensor) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
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
    rows = 2
    cols = 3

    # Load an image
    base_image = torchvision.io.read_image("examples/assets/english_springer.jpg")

    # Create multiple augmented images and add them to a grid
    fig = make_subplots(rows=rows, cols=cols, horizontal_spacing=0.05, vertical_spacing=0.05)
    for r in range(rows):
        for c in range(cols):
            # Augment and convert to numpy
            augmented = augment(base_image)
            augmented_hwc = augmented.permute(1, 2, 0)
            augmented_numpy = augmented_hwc.numpy()

            # Add to grid
            subfig = px.imshow(augmented_numpy)
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
            fig.add_trace(subfig.data[0], row=r + 1, col=c + 1)

    # Remove axes
    for r in range(rows):
        for c in range(cols):
            fig.update_xaxes(title_text="", showticklabels=False, showgrid=False, row=r + 1, col=c + 1)
            fig.update_yaxes(title_text="", showticklabels=False, showgrid=False, row=r + 1, col=c + 1)

    # Set height
    fig.update_layout(height=400)

    # Show the grid
    visual.print("Augmented images")
    visual.show(fig)
