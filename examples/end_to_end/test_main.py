from pathlib import Path
from typing import List

from torch import Tensor
from PIL import Image
import numpy as np

from examples.end_to_end.main import ClockDataset, ClockCoordinateDataset, Time, get_label, mean_norm, std_norm
from visual.interface import VisualFixture, fix_seeds, visual, standardize

test_data_path = Path("examples/end_to_end/test_data")


def test_ClockDataset(visual: VisualFixture, fix_seeds):
    dataset = ClockDataset(test_data_path / "train")
    images, labels = [], []
    for image, label in dataset:
        images.append(standardize(image.numpy(), mean_denorm=mean_norm, std_denorm=std_norm))
        labels.append(str(label))
    visual.show_images(images, labels)


#def test_ClockCoordinateDataset(visual: VisualFixture, fix_seeds):
#    dataset = ClockCoordinateDataset(test_data_path / "train")
#
#def visualize_dataset(visual: VisualFixture, dataset: ClockCoordinateDataset):
#    images: List[np.ndarray] = []
#    for image, label in dataset:
#        image = standardize(image.numpy(), mean_denorm=mean_norm, std_denorm=std_norm)
#        images.append(image)
#
#    visual.show_images(images)



def test_get_label():
    assert get_label(Path("examples/end_to_end/test_data/train/hash_00_00.png")) == Time(0, 0)
    assert get_label(Path("examples/end_to_end/test_data/train/hash_11_24.png")) == Time(11, 24)
