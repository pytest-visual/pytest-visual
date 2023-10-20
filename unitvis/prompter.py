import threading
import time
from typing import List, Optional

import dash_bootstrap_components as dbc
import plotly
import pytest
from dash import Dash, Input, Output, State, callback, ctx, dcc, html
from flask import Flask

from unitvis.io import Statement

plotly.io.templates.default = "plotly_white"
update_interval_ms = 200

_global_button_clicked: Optional[str] = None


@pytest.fixture(scope="session")
def unitvis_prompter() -> "Prompter":
    return Prompter()


class Prompter:
    def __init__(self) -> None:
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.layout = self._draw_initial_layout()

        self.thread = threading.Thread(target=self.app.run_server, kwargs={"debug": False, "use_reloader": False})
        self.thread.daemon = True
        self.thread.start()

        @self.app.callback(
            Output("accept-button", "n_clicks_timestamp"),  # Output is dummy, just to trigger the callback
            Output("decline-button", "n_clicks_timestamp"),  # Output is dummy, just to trigger the callback
            Input("accept-button", "n_clicks"),
            Input("decline-button", "n_clicks"),
        )
        def on_button_click(accept_clicks: int, decline_clicks: int) -> None:
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

        @self.app.callback(
            Output("prev-statements", "children"),
            Output("curr-statements", "children"),
            Input("interval-component", "n_intervals"),
        )
        def update_layout(n_intervals: int):
            prev_statements = self.app.layout["prev-statements"]
            curr_statements = self.app.layout["curr-statements"]
            return prev_statements.children, curr_statements.children

    def prompt_user(self, prev_statements: Optional[List[Statement]], curr_statements: List[Statement]) -> bool:
        if prev_statements is None:
            prev_statements = [["print", "No visualization cache"]]

        self._render_statements_in_div(prev_statements, "prev-statements")
        self._render_statements_in_div(curr_statements, "curr-statements")

        return self._get_accept_decline()

    def _draw_initial_layout(self) -> html.Div:
        """
        The layout contains accept/decline buttons at the top,
        and two horizontally separated divisions below it.
        The left division should contain the previous statements,
        and the right division should contain the current statements.
        """

        return html.Div(
            [
                dcc.Interval(id="interval-component", interval=update_interval_ms, n_intervals=0),
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

    def _render_statements_in_div(self, statements: List[Statement], div_id: str) -> None:
        rendered_statements: List[html.Div] = []
        for cmd, contents in statements:
            if cmd == "print":
                rendered_statements.append(html.Div(contents))
            elif cmd == "plot":
                figure = plotly.io.from_json(contents)
                rendered_statements.append(html.Div(dcc.Graph(figure=figure)))
            else:
                raise ValueError(f"Invalid command {cmd}")

        div = html.Div(rendered_statements, id=div_id)
        self.app.layout[div_id] = div

    def _get_accept_decline(self) -> bool:
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
                time.sleep(0.1)
