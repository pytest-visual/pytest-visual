from typing import Tuple

from _pytest.fixtures import FixtureRequest


def get_visualization_flags(request: FixtureRequest) -> Tuple[bool, bool, bool]:
    visualize = bool(request.config.getoption("--visual"))
    yes_all = bool(request.config.getoption("--visual-yes-all"))
    reset_all = bool(request.config.getoption("--visual-reset-all"))

    assert visualize + yes_all + reset_all <= 1, "Only one of --visual, --visual-yes-all, --visual-reset-all can be set"

    run_visualization = visualize or yes_all
    return run_visualization, yes_all, reset_all
