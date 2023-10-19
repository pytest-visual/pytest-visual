from plotly.graph_objs import Figure
from typing import List
from unitvis.io import Statement

class Visualize:
    def __init__(self):
        self.statements: List[Statement] = []

    def print(self, text) -> None:
        self.statements.append(["print", text])

    def show(self, fig: Figure) -> None:
        self.statements.append(["plot", str(fig.to_json())])
