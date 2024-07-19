import sys
from typing import Tuple

from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest


def print_visualization_message(port_number: int) -> None:
    """
    Prints a message to the console about the visualization server.

    Parameters:
    - port_number (int): The port number where the visualization server is running.
    """

    if "--visual" in sys.argv:
        print()
        print(f"-----  Visualizations will be shown at http://127.0.0.1:{port_number}  -----")
        print()


def pytest_addoption(parser: Parser):
    parser.addoption("--visual", action="store_true", help="Run visualization tests, prompt for acceptance")
    parser.addoption("--visual-accept-all", action="store_true", help="Visualization tests are accepted without prompting")
    parser.addoption("--visual-forget-all", action="store_true", help="Don't visualize, but mark all visualization cases as unaccepted")  # fmt: skip


def get_visualization_flags(request: FixtureRequest) -> Tuple[bool, bool, bool]:
    """
    Retrieves visualization flags from the pytest command line options. It checks which flags are set and ensures that they are mutually exclusive.

    Parameters:
    - request (FixtureRequest): A pytest request object containing configuration details and command line options.

    Returns:
    - Tuple[bool, bool, bool]: A tuple containing three boolean values corresponding to whether each flag is set:
        - run_visualization: True if visualization should be run (if --visual or --visual-accept-all is set).
        - accept_all: True if --visual-accept-all flag is set.
        - forget_all: True if --visual-forget-all flag is set.

    Raises:
    - AssertionError: If more than one of the flags is set.
    """

    return get_options(request.config)


def get_options(config: Config) -> Tuple[bool, bool, bool]:
    visualize = bool(config.getoption("--visual"))
    accept_all = bool(config.getoption("--visual-accept-all"))
    forget_all = bool(config.getoption("--visual-forget-all"))

    assert visualize + accept_all + forget_all <= 1, "Only one of --visual, --visual-accept-all, --visual-forget-all can be set"

    run_visualization = visualize or accept_all
    return run_visualization, accept_all, forget_all
