from typing import Tuple

from _pytest.fixtures import FixtureRequest


def get_visualization_flags(request: FixtureRequest) -> Tuple[bool, bool, bool]:
    """
    Retrieves visualization flags from the pytest command line options. It checks which flags are set and ensures that they are mutually exclusive.

    Parameters:
    - request (FixtureRequest): A pytest request object containing configuration details and command line options.

    Returns:
    - Tuple[bool, bool, bool]: A tuple containing three boolean values corresponding to whether each flag is set:
        - run_visualization: True if visualization should be run (if --visual or --visual-yes-all is set).
        - yes_all: True if --visual-yes-all flag is set.
        - reset_all: True if --visual-reset-all flag is set.

    Raises:
    - AssertionError: If more than one of the flags is set.
    """

    visualize = bool(request.config.getoption("--visual"))
    yes_all = bool(request.config.getoption("--visual-yes-all"))
    reset_all = bool(request.config.getoption("--visual-reset-all"))

    assert visualize + yes_all + reset_all <= 1, "Only one of --visual, --visual-yes-all, --visual-reset-all can be set"

    run_visualization = visualize or yes_all
    return run_visualization, yes_all, reset_all
