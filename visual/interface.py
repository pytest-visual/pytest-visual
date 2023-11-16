import random
from typing import Generator, List, Optional

import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest
from plotly.graph_objs import Figure

from visual.lib.convenience import (
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
    read_statements,
    write_statements,
)
from visual.lib.ui import UI, Location, visual_UI


class VisualFixture:
    def __init__(self):
        """
        An object to collect visualization statements.
        """
        self.statements: List[Statement] = []

    # Core interface

    def print(self, text: str) -> None:
        """
        Show text within a visualization case.

        Parameters:
        - text (str): The text to show.
        """
        self.statements.append(["print", text])

    def show(self, fig: Figure) -> None:
        """
        Show a plotly figure within a visualization case.

        Parameters:
        - fig (Figure): The figure to show.
        """
        self.statements.append(["show", str(fig.to_json())])

    # Convenience interface

    def show_images(
        self,
        images: List[np.ndarray],
        max_cols: int = 3,
        layout: Optional[str] = None,
        mean_denorm: Optional[List[float]] = None,
        std_denorm: Optional[List[float]] = None,
        min_value: float = 0,
        max_value: Optional[float] = None,
        height_per_row: int = 200,
    ) -> None:
        """
        Convenience method to show a grid of images. Accepts only numpy arrays, but supports a
        variety of shapes.

        Parameters:
        - images (List[np.ndarray]): A list of images to show.
        - max_cols (int): Maximum number of columns in the grid.
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
        - height_per_row (int): The height of each row in the grid.
        """
        assert all(isinstance(image, np.ndarray) for image in images), "Images must be numpy arrays"
        assert len(images) > 0, "At least one image must be specified"

        grid_shape = get_grid_shape(len(images), max_cols)
        layout = get_layout_from_image(layout, images[0])
        max_value = get_image_max_value_from_type(max_value, images[0])
        fig = create_plot_from_images(
            images, grid_shape, layout, mean_denorm, std_denorm, min_value, max_value, height_per_row * grid_shape[0]
        )
        self.show(fig)


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
        yield visualizer  # Run test

        statements = visualizer.statements

        if not yes_all:
            # Read previous statements
            location = Location(request.node.module.__file__, request.node.name)  # type: ignore
            prev_statements = read_statements(storage_path)

            # Check if statements have changed, and prompt user if so
            if statements != prev_statements:
                if not visual_UI.prompt_user(location, prev_statements, statements):
                    pytest.fail("Visualizations were not accepted")

        # No declined changes or --visual-yes-all flag set, so accept changes
        write_statements(storage_path, statements)
    elif reset_all:
        # Reset visualization
        clear_statements(storage_path)
        pytest.skip("Resetting visualization case as per --visualize-reset-all")
    else:
        pytest.skip("Visualization is not enabled, add --visual option to enable")


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
