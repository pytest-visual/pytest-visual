from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest
from PIL import Image
from torch import Tensor

from examples.end_to_end.main import (
    ClockCoordinateDataset,
    ClockDataset,
    Time,
    get_label,
    mean_norm,
    std_norm,
)
from visual.interface import VisualFixture, fix_seeds, standardize, visual

test_data_path = Path("examples/end_to_end/test_data")


def test_original_labels(visual: VisualFixture, fix_seeds):
    dataset = ClockDataset(test_data_path / "train")
    images, labels = [], []
    for image, label in dataset:
        images.append(standardize(image.numpy(), mean_denorm=mean_norm, std_denorm=std_norm))
        labels.append(str(label))

    visual.print("Dataset with time labels:")
    visual.show_images(images, labels)


def test_with_hands(visual: VisualFixture, fix_seeds):
    dataset_non_aug = ClockCoordinateDataset(test_data_path / "train")
    visual.print("Dataset with hands drawn:")
    visualize_dataset(visual, dataset_non_aug)


def test_augmented_with_hands(visual: VisualFixture, fix_seeds):
    dataset_aug = ClockCoordinateDataset(test_data_path / "train", augment=True)
    visual.print("Augmented dataset with hands drawn:")
    visualize_dataset(visual, dataset_aug)


def visualize_dataset(visual: VisualFixture, dataset: ClockCoordinateDataset):
    images: List[np.ndarray] = []
    for image, label in dataset:
        image = standardize(image.numpy(), mean_denorm=mean_norm, std_denorm=std_norm)

        # Draw clock hands as lines
        image = image.copy()
        for hand_name in label:
            color = {"minute": (0, 255, 0), "hour": (255, 0, 0)}[hand_name]
            length = {"minute": 0.4, "hour": 0.2}[hand_name]

            coords = label[hand_name].numpy()
            center = np.array([image.shape[0] / 2, image.shape[1] / 2])
            end = center + length * coords * image.shape[0]
            cv2.line(image, tuple(center.astype(int)), tuple(end.astype(int)), color, 1)

        images.append(image)

    visual.show_images(images)


def test_get_label():
    assert get_label(Path("examples/end_to_end/test_data/train/hash_00_00.png")) == Time(0, 0)
    assert get_label(Path("examples/end_to_end/test_data/train/hash_11_24.png")) == Time(11, 24)


def test_back_and_forth_conversion():
    for hour in range(12):
        for minute in range(60):
            t = Time(hour, minute)
            coords = t.get_coords()
            t_reconstructed = Time.from_coords(coords)
            assert t == t_reconstructed


def test_percent_from_xy():
    assert Time.percent_from_xy(Tensor([0, -1])) == pytest.approx(0)
    assert Time.percent_from_xy(Tensor([1, 0])) == pytest.approx(0.25)
    assert Time.percent_from_xy(Tensor([0, 1])) == pytest.approx(0.5)
    assert Time.percent_from_xy(Tensor([-1, 0])) == pytest.approx(0.75)
