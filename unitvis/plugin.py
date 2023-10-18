import pytest
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from typing import Tuple

from unitvis.visualize import Visualize

def pytest_addoption(parser: Parser):
    parser.addoption("--visualize", action="store_true", help="Run visualization tests, prompt for acceptance")
    parser.addoption("--visualize-yes-all", action="store_true", help="Visualization tests are accepted without prompting")
    parser.addoption("--visualize-reset-all", action="store_true", help="Don't visualize, but mark all visualization cases as unaccepted")


@pytest.fixture
def visualize(request: FixtureRequest):
    run_visualization, yes_all, reset_all = get_visualization_flags(request)
    visualizer = Visualize()
    if run_visualization:
        # Run the case
        yield visualizer

        if yes_all:
            visualizer.accept_all()
        else:
            visualizer.prompt()
    elif reset_all:
        visualizer.reset_all()
        pytest.skip("Resetting visualization case as per --visualize-reset-all")
    else:
        pytest.skip("Visualization is not enabled, add --visualize option to enable")



def get_visualization_flags(request: FixtureRequest) -> Tuple[bool, bool, bool]:
    visualize = bool(request.config.getoption("--visualize"))
    yes_all = bool(request.config.getoption("--visualize-yes-all"))
    reset_all = bool(request.config.getoption("--visualize-reset-all"))

    assert visualize + yes_all + reset_all <= 1, "Only one of --visualize, --visualize-yes-all, --visualize-reset-all can be set"

    run_visualization = visualize or yes_all
    return run_visualization, yes_all, reset_all