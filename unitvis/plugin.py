import pytest
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from typing import Tuple, List
from pathlib import Path

from plotly.graph_objs import Figure
from unitvis.io import get_storage_path, read_statements, write_statements, clear_statements, Statement
from unitvis.prompter import unitvis_prompter, Prompter

def pytest_addoption(parser: Parser):
    parser.addoption("--visualize", action="store_true", help="Run visualization tests, prompt for acceptance")
    parser.addoption("--visualize-yes-all", action="store_true", help="Visualization tests are accepted without prompting")
    parser.addoption("--visualize-reset-all", action="store_true", help="Don't visualize, but mark all visualization cases as unaccepted")


@pytest.fixture
def visualize(request: FixtureRequest, unitvis_prompter: Prompter):
    run_visualization, yes_all, reset_all = _get_visualization_flags(request)
    visualizer = Visualize()
    storage_path = get_storage_path(request)

    if run_visualization:
        # Run test
        yield visualizer

        # Handle visualization
        statements = visualizer.statements
        if yes_all:
            _teardown_with_yes_all(storage_path, statements)
        else:
            _teardown_with_verification(unitvis_prompter, storage_path, statements)
    elif reset_all:
        _teardown_with_reset_all(storage_path)
    else:
        pytest.skip("Visualization is not enabled, add --visualize option to enable")


def _get_visualization_flags(request: FixtureRequest) -> Tuple[bool, bool, bool]:
    visualize = bool(request.config.getoption("--visualize"))
    yes_all = bool(request.config.getoption("--visualize-yes-all"))
    reset_all = bool(request.config.getoption("--visualize-reset-all"))

    assert visualize + yes_all + reset_all <= 1, "Only one of --visualize, --visualize-yes-all, --visualize-reset-all can be set"

    run_visualization = visualize or yes_all
    return run_visualization, yes_all, reset_all


def _teardown_with_yes_all(path: Path, statements: List[Statement]) -> None:
    write_statements(path, statements)


def _teardown_with_verification(prompter: Prompter, path: Path, statements: List[Statement]) -> None:
    prev_statements = read_statements(path)

    if statements != prev_statements:
        if prompter.prompt_user(prev_statements, statements):
            write_statements(path, statements)
        else:
            pytest.fail("Visualizations were not accepted")


def _teardown_with_reset_all(path: Path) -> None:
    clear_statements(path)
    pytest.skip("Resetting visualization case as per --visualize-reset-all")


class Visualize:
    def __init__(self):
        self.statements: List[Statement] = []

    def print(self, text) -> None:
        self.statements.append(["print", text])

    def show(self, fig: Figure) -> None:
        self.statements.append(["plot", str(fig.to_json())])
