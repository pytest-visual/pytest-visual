import logging
import threading
import time
from typing import Generator, List, Optional

import dash_bootstrap_components as dbc
import plotly
import pytest
from _pytest.fixtures import FixtureRequest
from dash import Dash, Input, Output, ctx, dcc, html
from plotly.graph_objs import Figure

from visual.lib.flags import get_visualization_flags, print_visualization_message
from visual.lib.storage import Statement

logging.basicConfig(level=logging.INFO)  # To see Dash url

port_number = 54545
plotly.io.templates.default = "plotly_white"

# In seconds
accept_decline_polling_interval = 0.1
update_interval = 0.5
finish_delay = 1.0

_global_button_clicked: Optional[str] = None

print_visualization_message(port_number)


@pytest.fixture(scope="session")
def visual_UI(request: FixtureRequest) -> Generator[Optional["UI"], None, None]:
    """
    A pytest fixture that conditionally sets up and tears down a UI object for visualization purposes during testing.

    The decision to yield a UI object or None is based on flags obtained from the test session configuration.
    The UI object, if created, runs a Dash server in a separate thread for the duration of the test session.
    """
    run_visualization, accept_all, forget_all = get_visualization_flags(request)
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

        self.thread = threading.Thread(
            target=self.app.run_server, kwargs={"debug": False, "use_reloader": False, "port": port_number}
        )
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
                dbc.Card(
                    [
                        dbc.ListGroup(
                            [
                                dbc.CardHeader("", id="file-name"),
                                dbc.CardGroup(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.Button(
                                                    "Decline",
                                                    style={
                                                        "backgroundColor": "#d9534f",
                                                        "margin": "10px",
                                                        "width": "100% - 20px",
                                                    },
                                                    id="decline-button",
                                                    n_clicks=0,
                                                ),  # fmt: skip
                                                dbc.CardBody(id="prev-statements"),
                                                dbc.CardFooter("Previously accepted", style={"textAlign": "center"}),
                                            ],
                                        ),
                                        dbc.Card(
                                            [
                                                dbc.Button(
                                                    "Accept",
                                                    style={
                                                        "backgroundColor": "#5cb85c",
                                                        "margin": "10px",
                                                        "width": "100% - 20px",
                                                    },
                                                    id="accept-button",
                                                    n_clicks=0,
                                                ),  # fmt: skip
                                                dbc.CardBody(id="curr-statements"),
                                                dbc.CardFooter("New", style={"textAlign": "center"}),
                                            ],
                                        ),
                                    ],
                                    style={"display": "flex", "margin": "10px"},
                                ),
                            ],
                            flush=True,
                            style={"border": "1px solid #ccc"},
                        ),
                    ],
                    style={"padding": "10px", "border": "0px"},
                ),
            ],
        )

    def _render_location(self, location: Optional["Location"]) -> None:
        """
        Renders the location of the test function into the UI.
        """
        if location is None:
            location = Location("", "")

        element = [
            f"File: {location.file_name}",
            html.Br(),
            f"Function: {location.function_name + '()' if location.function_name != '' else ''}",
        ]
        self.file_name = dbc.CardHeader(element, id="file-name")

    def _render_statements_in_div(self, statements: Optional[List[Statement]], div_id: str) -> dbc.CardBody:
        """
        Renders statements into a specified division in the UI.
        Each statement could either be a text statement, plotly figure, or images.
        """

        code_style = {
            "background-color": "#f8f8f8",
            "border": "1px solid #999",
            "display": "block",
            "padding": "10px",
            "border-radius": "5px",
            "white-space": "pre-wrap",
            "margin-top": "10px",
            "margin-bottom": "10px",
            "font-family": "monospace",
            "color": "black",
        }
        plot_style = {"padding": "10px", "margin-top": "10px", "margin-bottom": "10px"}

        rendered_statements: list = []
        if statements is None:
            rendered_statements.append(html.P("Nothing to show"))
        else:
            for statement in statements:
                if statement.Type == "text":
                    rendered_statements.append(html.Code(statement.Text, style=code_style))
                elif statement.Type == "figure":
                    assert type(statement.Asset) == Figure, "A figure statement must have a Figure asset"
                    rendered_statements.append(dbc.Card(dcc.Graph(figure=statement.Asset), style=plot_style))
                else:
                    raise ValueError(f"Invalid command {statement.Type}")

        div = dbc.CardBody(rendered_statements, id=div_id)
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
