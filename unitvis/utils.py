
from typing import Tuple
from _pytest.fixtures import FixtureRequest


def get_visualization_flags(request: FixtureRequest) -> Tuple[bool, bool, bool]:
    visualize = bool(request.config.getoption("--visualize"))
    yes_all = bool(request.config.getoption("--visualize-yes-all"))
    reset_all = bool(request.config.getoption("--visualize-reset-all"))

    assert visualize + yes_all + reset_all <= 1, "Only one of --visualize, --visualize-yes-all, --visualize-reset-all can be set"

    run_visualization = visualize or yes_all
    return run_visualization, yes_all, reset_all
