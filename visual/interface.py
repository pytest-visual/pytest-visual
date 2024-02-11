import os
import random
import tempfile
from typing import Generator, List, Optional

import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest
from PIL import Image
from plotly.graph_objs import Figure

from visual.lib.convenience import (
    correct_layout,
    create_plot_from_images,
    get_grid_shape,
    get_image_max_value_from_type,
    get_layout_from_image,
)
from visual.lib.flags import get_visualization_flags, pytest_addoption
from visual.lib.storage import (
    Statement,
    clear_statements,
    get_storage_path,
    load_statements,
    store_statements,
)
from visual.lib.ui import UI, Location, visual_UI


class VisualFixture:
    def __init__(self):
        """
        An object to collect visualization statements.
        """
        self.statements: List[Statement] = []

    # Core interface

    def text(self, text: str) -> None:
        """
        Show text within a visualization case.

        Parameters:
        - text (str): The text to show.
        """
        self.statements.append(["text", text])

    def figure(self, figure: Figure) -> None:
        """
        Show a plotly figure within a visualization case.

        Parameters:
        - fig (Figure): The figure to show.
        """
        self.statements.append(["figure", str(figure.to_json())])

    def images(
        self,
        images: List[np.ndarray],
        labels: Optional[List[str]] = None,
        max_cols: int = 3,
        image_size: float = 300,
    ) -> None:
        """
        Convenience method to show a grid of images. Only accepts standardized numpy images.

        Parameters:
        - images (List[np.ndarray]): A list of images to show.
        - labels (Optional[List[str]]): A list of labels for each image.
        - max_cols (int): Maximum number of columns in the grid.
        - image_size (float): The larger side of each image in the grid, in pixels.
        """
        assert all(isinstance(image, np.ndarray) for image in images), "Images must be numpy arrays"
        assert len(images) > 0, "At least one image must be specified"

        grid_shape = get_grid_shape(len(images), max_cols)
        total_height = None if image_size is None else image_size * grid_shape[0]

        figure = create_plot_from_images(images, labels, grid_shape, total_height)
        self.figure(figure)

    # Convenience interface

    def image(
        self,
        image: np.ndarray,
        label: Optional[str] = None,
        image_size: float = 600,
    ) -> None:
        """
        Convenience method to show a single image. Only accepts standardized numpy images.

        Parameters:
        - image (np.ndarray): The image to show.
        - label (Optional[str]): A label for the image.
        - image_size (float): The height of the image.
        """
        labels = None if label is None else [label]
        self.images([image], labels, max_cols=1, image_size=image_size)

    def model(
        self,
        model,
        input_size,
        depth: int = 100,
        image_size: float = 1500,
    ) -> None:
        """
        Convenience method to show a PyTorch model. Requires the torchview package.

        Parameters:
        - model (torch.nn.Module): The model to show.
        - input_size (Tuple[int, ...]): The input size of the model.
        - depth (int): The maximum depth of the model to show.
        - image_size (float): The height of the image.
        """
        import torchview  # isort: skip

        plot = torchview.draw_graph(model, input_size=input_size, depth=depth)

        # Create temporary file path
        tempfile_path = tempfile.mktemp()
        plot.visual_graph.render(tempfile_path, format="png")

        # Read image and show
        image = np.array(Image.open(tempfile_path + ".png"))
        self.image(image, image_size=image_size)

        # Remove temporary file
        os.remove(tempfile_path)
        os.remove(tempfile_path + ".png")


@pytest.fixture
def visual(request: FixtureRequest, visual_UI: UI) -> Generator[VisualFixture, None, None]:
    """
    A pytest fixture that manages the visualization process during test execution.

    Parameters:
    - request (FixtureRequest): The current pytest request.
    - visual_UI (UI): An instance of the UI class for user interaction.

    Yields:
    - VisualFixture: An object to collect visualization statements.
    """
    run_visualization, yes_all, reset_all = get_visualization_flags(request)
    visualizer = VisualFixture()
    storage_path = get_storage_path(request)

    if run_visualization:
        failed_tests1 = request.session.testsfailed
        yield visualizer  # Run test
        failed_tests2 = request.session.testsfailed

        if failed_tests2 > failed_tests1:
            return  # Test failed, so no visualization

        statements = visualizer.statements

        if not yes_all:
            # Read previous statements
            location = Location(request.node.module.__file__, request.node.name)  # type: ignore
            prev_statements = load_statements(storage_path)

            # Check if statements have changed, and prompt user if so
            if statements != prev_statements:
                if not visual_UI.prompt_user(location, prev_statements, statements):
                    pytest.fail("Visualizations were not accepted")

        # No declined changes or --visual-yes-all flag set, so accept changes
        store_statements(storage_path, statements)
    elif reset_all:
        # Reset visualization
        clear_statements(storage_path)
        pytest.skip("Resetting visualization case as per --visualize-reset-all")
    else:
        pytest.skip("Visualization is not enabled, add --visual option to enable")


# Convenience features


@pytest.fixture
def fix_seeds() -> None:
    """
    A pytest fixture that fixes the random seeds of random, and optionally numpy, torch and tensorflow.
    """
    random.seed(0)

    try:
        import numpy as np

        np.random.seed(0)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore
    except (ImportError, AttributeError):
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(0)
    except ImportError:
        pass


def standardize(
    image: np.ndarray,
    layout: Optional[str] = None,
    mean_denorm: Optional[List[float]] = None,
    std_denorm: Optional[List[float]] = None,
    min_value: float = 0,
    max_value: Optional[float] = None,
) -> np.ndarray:
    """
    - layout (Optional[str]): The shape of the images. If not specified, the shape is
        determined automatically. Supported shapes are "hwc", "chw", "hw", "1chw", "1hwc".
    - mean_comp (Optional[List[float]]): The mean that was used to normalize the images, which
        is used to denormalize the images. If not specified, the images are not denormalized.
    - std_comp (Optional[List[float]]): The standard deviation that was used to normalize the
        images, which is used to denormalize the images. If not specified, the images are not
        denormalized.
    - min_value (float): The assumed minimum value of the images.
    - max_value (Optional[float]): The assumed maximum value of the images. If not specified,
        the maximum value is 1 for float images and 255 for integer images.
    """

    # Get layout and max value
    layout = get_layout_from_image(layout, image)
    max_value = get_image_max_value_from_type(max_value, image)

    # Denormalize, convert to uint8, and correct layout
    image = correct_layout(image, layout)
    if std_denorm is not None:
        image = image * np.array(std_denorm)
    if mean_denorm is not None:
        image = image + mean_denorm
    image = (image - min_value) / (max_value - min_value) * 255
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image
