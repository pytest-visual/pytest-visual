import math
import random
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

data_dir = Path("data")
models_dir = Path("models")
mean_norm, std_norm = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def main() -> None:
    # Load data
    train_dataset = ClockCoordinateDataset(data_dir / "train", augment=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_dataset = ClockCoordinateDataset(data_dir / "val")
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_dataset = ClockDataset(data_dir / "val")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)

    # Train and test
    train(model, train_loader, val_loader, device)
    test(model, test_loader, device)
    torch.save(model.state_dict(), models_dir / "model.pth")


# Data loading


class ClockDataset(Dataset[Tuple[Tensor, "Time"]]):
    """
    Dataset for clock images. The labels are Time objects representing the time shown on the clock.
    """

    def __init__(self, data_dir: Path):
        super().__init__()
        self.paths = []
        for pth in data_dir.glob("*"):
            if pth.is_file():
                self.paths.append(pth)
        self.paths = sorted(self.paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, "Time"]:
        path = self.paths[index]

        image = Image.open(path)
        image_tensor = F.to_image(image).to(torch.float32) / 255  # type: ignore
        image_tensor = normalize_image(image_tensor)
        label = get_label(path)

        return image_tensor, label

    def __len__(self) -> int:
        return len(self.paths)


class ClockCoordinateDataset(Dataset[Tuple[Tensor, Tensor]]):
    """
    Dataset for clock images. The labels are Tensors of shape (4,) representing the coordinates of the hour and minute hands.
    """

    def __init__(self, data_dir: Path, augment: bool = False):
        super().__init__()
        self.clock_dataset = ClockDataset(data_dir)
        self.augment = augment

    def __getitem__(self, index: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        image, label = self.clock_dataset[index]
        coords = label.get_coords()
        if self.augment:
            image, coords = augment(image, coords)
        return image, coords
    
    def __len__(self) -> int:
        return len(self.clock_dataset)


class Time:
    def __init__(self, hour: int, minute: int):
        self.hour = hour
        self.minute = minute

    @staticmethod
    def from_coords(coords: Tensor) -> "Time":
        hour_xy, minute_xy = coords.reshape(2, 2)

        # Calculate minutes
        minute_percent = math.atan2(minute_xy[0], minute_xy[1]) / (2 * math.pi)
        minute = round(60 * minute_percent) % 60

        # Calculate hours, taking into account that the hour hand moves as the minute hand moves
        hour_percent = math.atan2(hour_xy[0], hour_xy[1]) / (2 * math.pi)
        hour_percent -= minute_percent / 12
        hour = round(12 * hour_percent) % 12

        return Time(hour, minute)

    def get_coords(self) -> Dict[str, Tensor]:
        # Return a tensor of shape (4,) with the coordinates of the hour and minute hands.
        hour_angle = 2 * math.pi * (self.hour % 12) / 12
        hour_xy = (math.sin(hour_angle), math.cos(hour_angle))
        minute_angle = 2 * math.pi * self.minute / 60
        minute_xy = (math.sin(minute_angle), math.cos(minute_angle))
        return {"hours": Tensor(hour_xy), "minutes": Tensor(minute_xy)}

    def approx_eq(self, other: "Time", max_diff_minutes: int = 5) -> bool:
        first = self.hour * 60 + self.minute
        second = other.hour * 60 + other.minute
        return abs(first - second) <= max_diff_minutes or abs(first - second) >= 24 * 60 - max_diff_minutes

    def __repr__(self) -> str:
        return f"{self.hour:02}:{self.minute:02}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Time):
            return False
        return self.hour == other.hour and self.minute == other.minute


def normalize_image(image: Tensor) -> Tensor:
    return F.normalize(image, mean=mean_norm, std=std_norm)


def get_label(path: Path) -> Time:
    # Load label from file name: <random_hash>_<hour>_<minute>.jpg
    filename = path.stem
    random_hash, hour_str, minute_str = filename.split("_")
    hour, minute = int(hour_str), int(minute_str)
    label = Time(hour, minute)
    return label


# Data augmentation


def augment(image: Tensor, label: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
    hour_xy, minute_xy = label["hours"], label["minutes"]

    # Randomly flip horizontally
    if random.random() > 0.5:
        image = F.hflip(image)
        hour_xy[0] = -hour_xy[0]
        minute_xy[0] = -minute_xy[0]

    # Randomly rotate and scale (note that only rotation affects the coordinates)
    angle = random.uniform(0, 360)
    scale = random.uniform(0.8, 1.2)
    image = F.affine(image, angle=angle, translate=[0, 0], scale=scale, shear=[0], interpolation=Image.BILINEAR)

    radians = math.radians(angle)
    rotation_matrix = Tensor([[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]])
    hour_xy = torch.mv(rotation_matrix, hour_xy)
    minute_xy = torch.mv(rotation_matrix, minute_xy)

    # Randomly adjust brightness
    brightness_factor = random.uniform(0.7, 1.4)
    image = F.adjust_brightness(image, brightness_factor)

    # Randomly adjust contrast
    contrast_factor = random.uniform(0.7, 1.4)
    image = F.adjust_contrast(image, contrast_factor)

    # Randomly adjust saturation
    saturation_factor = random.uniform(0.8, 1.2)
    image = F.adjust_saturation(image, saturation_factor)

    return image, {"hours": hour_xy, "minutes": minute_xy}


# Model


def get_model(pretrained: bool = True) -> nn.Module:
    model = resnet18(pretrained=pretrained)
    model.fc = get_model_head(model.fc.in_features)
    return model


def get_model_head(in_features: int) -> nn.Module:
    return nn.Linear(in_features, 4)


# Training


def train(
    model: nn.Module,
    train_loader: DataLoader[Tuple[Tensor, Tensor]],
    val_loader: DataLoader[Tuple[Tensor, Tensor]],
    device: torch.device,
) -> Tuple[List[float], List[float]]:
    # Setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train and validation epochs
    train_losses: List[float] = []
    val_losses: List[float] = []
    for epoch in range(10):
        train_losses.append(train_loop(model, train_loader, criterion, optimizer, device, epoch))
        val_losses.append(train_loop(model, val_loader, criterion, optimizer, device, epoch, train=False))

    return train_losses, val_losses


def train_loop(
    model: nn.Module,
    loader: DataLoader[Tuple[Tensor, Tensor]],
    criterion: nn.MSELoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    train: bool = True,
) -> float:
    model.train(mode=train)

    losses: List[float] = []
    with torch.set_grad_enabled(train):
        for image, label in loader:
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            losses.append(loss.cpu().item())
            if train:
                loss.backward()
                optimizer.step()

    print(f"Epoch {epoch} ({'train' if train else 'validation'}): {sum(losses) / len(losses) * 100:.2f}%")
    return sum(losses) / len(losses)


# Testing


def test(model: nn.Module, loader: DataLoader[Tuple[Tensor, Time]], device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for image, label_batch in loader:
        image, label_batch = image.to(device), label_batch.to(device)
        output_batch = model(image)
        for output, label in zip(output_batch, label_batch):
            output_time = Time.from_coords(output)
            if output_time.approx_eq(label):
                correct += 1
            total += 1

    print(f"Test accuracy: {correct / total * 100:.2f}%")
    return correct / total


if __name__ == "__main__":
    main()
