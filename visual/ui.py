import logging
import threading
import time
from typing import Generator, List, Optional

import dash_bootstrap_components as dbc
import plotly
import pytest
from _pytest.fixtures import FixtureRequest
from dash import Dash, Input, Output, ctx, dcc, html

from visual.storage import Statement
from visual.utils import get_visualization_flags

logging.basicConfig(level=logging.INFO)  # To see Dash url

plotly.io.templates.default = "plotly_white"

# In seconds
accept_decline_polling_interval = 0.1
update_interval = 0.2
finish_delay = 1.0


_global_button_clicked: Optional[str] = None


@pytest.fixture(scope="session")
def visual_UI(request: FixtureRequest) -> Generator[Optional["UI"], None, None]:
    """
    A pytest fixture that conditionally sets up and tears down a UI object for visualization purposes during testing.

    The decision to yield a UI object or None is based on flags obtained from the test session configuration.
    The UI object, if created, runs a Dash server in a separate thread for the duration of the test session.
    """
    run_visualization, yes_all, reset_all = get_visualization_flags(request)
    if run_visualization:  # Yield a UI object
        ui = UI()
        yield ui
        ui.teardown()
    else:  # No visualizations will be shown, so no need to start the heavy Dash server
        yield None


class UI:
    def __init__(self) -> None:
        """
        Initializes the UI object, setting up the app, the layout, and starting a thread to run the server.
        The callbacks for the interactive elements in the UI are also defined within this method.
        """
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.layout = self._draw_initial_layout()

        self._render_blank()

        self.thread = threading.Thread(target=self.app.run_server, kwargs={"debug": False, "use_reloader": False})
        self.thread.daemon = True
        self.thread.start()

        @self.app.callback(
            Output("accept-button", "n_clicks_timestamp"),  # Output is dummy, just to trigger the callback
            Output("decline-button", "n_clicks_timestamp"),  # Output is dummy, just to trigger the callback
            Input("accept-button", "n_clicks"),
            Input("decline-button", "n_clicks"),
        )
        def on_button_click(accept_clicks: int, decline_clicks: int) -> tuple:
            """
            Callback function that is triggered when either the 'Accept' or 'Decline' button is clicked.
            It modifies a global variable to reflect which button was clicked.
            """
            global _global_button_clicked

            if ctx.triggered_id == "accept-button":
                _global_button_clicked = "accept"
            elif ctx.triggered_id == "decline-button":
                _global_button_clicked = "decline"
            elif ctx.triggered_id is None:
                pass  # On reload, no button is clicked
            else:
                # Raised exceptions are not shown in the console, but prints are
                print(f"Invalid trigger: {ctx.triggered_id}")
            return None, None

        @self.app.callback(
            Output("file-name", "children"),
            Output("function-name", "children"),
            Output("prev-statements", "children"),
            Output("curr-statements", "children"),
            Input("interval-component", "n_intervals"),
        )
        def update_layout(n_intervals: int):
            """
            Callback function that updates the layout at specified intervals.
            This function keeps the UI updated in real-time.
            """
            return (
                self.file_name.children,  # type: ignore
                self.function_name.children,  # type: ignore
                self.prev_statements.children,
                self.curr_statements.children,
            )

    def teardown(self) -> None:
        """
        Renders a blank UI and waits for the layout to update.
        """
        self._render_blank()
        time.sleep(finish_delay)  # Wait for the layout to update

    def _render_blank(self) -> None:
        """
        Renders a blank UI, without any statements or buttons.
        """
        self._render_location(None)
        self.prev_statements = self._render_statements_in_div([], "prev-statements")
        self.curr_statements = self._render_statements_in_div([], "curr-statements")

    def prompt_user(
        self, location: "Location", prev_statements: Optional[List[Statement]], curr_statements: List[Statement]
    ) -> bool:
        """
        Prompts the user with statements for review and waits for user interaction (accept/decline).
        """
        self._render_location(location)

        if prev_statements is None:
            prev_statements = [["print", "No visualization cache"]]

        self.prev_statements = self._render_statements_in_div(prev_statements, "prev-statements")
        self.curr_statements = self._render_statements_in_div(curr_statements, "curr-statements")

        return self._get_accept_decline()

    def _draw_initial_layout(self) -> html.Div:
        """
        Creates and returns the initial layout of the app, organizing the visual elements.
        """

        return html.Div(
            [
                dcc.Interval(id="interval-component", interval=update_interval * 1000, n_intervals=0),
                html.Div(
                    [
                        html.Div(
                            [
                                dbc.Button("Decline", id="decline-button", n_clicks=0, style={"backgroundColor": "#d9534f"}),
                            ],
                            className="col",
                        ),
                        html.Div(
                            [
                                dbc.Button("Accept", id="accept-button", n_clicks=0, style={"backgroundColor": "#5cb85c"}),
                            ],
                            className="col",
                        ),
                    ],
                    style={"textAlign": "center"},
                    className="row",
                ),
                html.Div(
                    [
                        html.H4("", id="file-name"),
                        html.H4("", id="function-name"),
                    ],
                    style={"paddingBottom": "20px", "paddingTop": "20px"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H5("Previously accepted"),
                                html.Div(id="prev-statements"),
                            ],
                            style={"flex": 1, "marginRight": "10px", "padding": "10px", "border": "1px solid #ccc"},
                        ),
                        html.Div(
                            [
                                html.H5("New"),
                                html.Div(id="curr-statements"),
                            ],
                            style={"flex": 1, "marginLeft": "10px", "padding": "10px", "border": "1px solid #ccc"},
                        ),
                    ],
                    style={"display": "flex", "margin": "10px"},
                    className="row",
                ),
            ],
            style={"padding": "10px"},
            className="container",
        )

    def _render_location(self, location: Optional["Location"]) -> None:
        """
        Renders the location of the test function into the UI.
        """
        if location is None:
            location = Location("", "")

        print("LOCATION FUNCTION AND FILE NAME:", location.function_name, location.file_name)
        self.file_name = html.H4(f"File: {location.file_name}", id="file-name")
        self.function_name = html.H4(f"Function: {location.function_name + '()' if location.function_name != '' else ''}", id="function-name")  # fmt: skip

    def _render_statements_in_div(self, statements: List[Statement], div_id: str) -> html.Div:
        """
        Renders statements into a specified division in the UI.
        Each statement could either be a print statement or a graphical (plotly) figure.
        """

        rendered_statements: List[html.Div] = []
        for cmd, contents in statements:
            if cmd == "print":
                rendered_statements.append(html.Div(contents))
            elif cmd == "show":
                figure = plotly.io.from_json(contents)
                rendered_statements.append(html.Div(dcc.Graph(figure=figure)))
            else:
                raise ValueError(f"Invalid command {cmd}")

        div = html.Div(rendered_statements, id=div_id)
        return div

    def _get_accept_decline(self) -> bool:
        """
        Waits for the user to click either 'Accept' or 'Decline', and returns a boolean value reflecting the choice.
        """
        global _global_button_clicked

        while True:
            if _global_button_clicked is not None:
                if _global_button_clicked == "accept":
                    _global_button_clicked = None
                    return True
                elif _global_button_clicked == "decline":
                    _global_button_clicked = None
                    return False
                else:
                    _global_button_clicked = None
                    raise ValueError("Invalid button clicked value")
            else:
                time.sleep(accept_decline_polling_interval)


class Location:
    def __init__(self, file_name: str, function_name: str) -> None:
        self.file_name = file_name
        self.function_name = function_name
