from pathlib import Path
from typing import Generator, List

import pytest
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from plotly.graph_objs import Figure

from visual.storage import (
    Statement,
    clear_statements,
    get_storage_path,
    read_statements,
    write_statements,
)
from visual.ui import UI, Location, visual_UI
from visual.utils import get_visualization_flags


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


def pytest_addoption(parser: Parser):
    parser.addoption("--visual", action="store_true", help="Run visualization tests, prompt for acceptance")
    parser.addoption("--visual-yes-all", action="store_true", help="Visualization tests are accepted without prompting")
    parser.addoption("--visual-reset-all", action="store_true", help="Don't visualize, but mark all visualization cases as unaccepted")  # fmt: skip


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
        if yes_all:
            _accept_changes(storage_path, statements)
        else:
            location = Location(request.node.module.__file__, request.node.name)
            _query_user_for_acceptance(location, visual_UI, storage_path, statements)
    elif reset_all:
        _clear_cache(storage_path)
    else:
        pytest.skip("Visualization is not enabled, add --visual option to enable")


def _accept_changes(path: Path, statements: List[Statement]) -> None:
    write_statements(path, statements)


def _query_user_for_acceptance(location: Location, prompter: UI, path: Path, statements: List[Statement]) -> None:
    prev_statements = read_statements(path)

    if statements != prev_statements:
        if prompter.prompt_user(location, prev_statements, statements):
            write_statements(path, statements)
        else:
            pytest.fail("Visualizations were not accepted")


def _clear_cache(path: Path) -> None:
    clear_statements(path)
    pytest.skip("Resetting visualization case as per --visualize-reset-all")
