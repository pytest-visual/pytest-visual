from pathlib import Path

from torch import Tensor

from examples.end_to_end.main import ClockDataset, Time, mean_norm, std_norm
from visual.interface import VisualFixture, fix_seeds, visual

test_data_path = Path("examples/end_to_end/test_data")


def test_ClockDataset(visual: VisualFixture, fix_seeds):
    dataset = ClockDataset(test_data_path / "train")
    images, labels = [], []
    print(len(dataset))
    for i in range(len(dataset)):
        image, label = dataset[i]
        print(image[0, 0, 0])
        images.append(image.numpy())
        labels.append(str(label))
    visual.show_images(images, labels, mean_denorm=mean_norm, std_denorm=std_norm)
