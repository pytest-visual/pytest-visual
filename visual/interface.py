import random
from typing import Generator, List

import pytest
from _pytest.fixtures import FixtureRequest
from plotly.graph_objs import Figure

from visual.lib.flags import get_visualization_flags, pytest_addoption
from visual.lib.storage import (
    Statement,
    clear_statements,
    get_storage_path,
    read_statements,
    write_statements,
)
from visual.lib.ui import UI, Location, visual_UI

# Core interface


class VisualFixture:
    def __init__(self):
        """
        Initializer for the VisualFixture class which collects print and show statements during a test.
        These statements can be stored, loaded, compared, and visualized.
        """
        self.statements: List[Statement] = []

    def print(self, text) -> None:
        self.statements.append(["print", text])

    def show(self, fig: Figure) -> None:
        self.statements.append(["show", str(fig.to_json())])


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


# High level interface


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
