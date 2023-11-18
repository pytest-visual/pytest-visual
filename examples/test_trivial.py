import plotly.express as px

from visual.interface import VisualFixture


def test_show_square(visual: VisualFixture):
    x, y = [0, 1, 2, 3, 4, 5], [0, 1, 4, 9, 16, 25]
    fig = px.scatter(x=x, y=y)

    visual.print("Square plot")
    visual.show_figure(fig)


def test_show_square_root(visual: VisualFixture):
    x, y = [0, 1, 2, 3, 4, 5], [0, 1**0.5, 2**0.5, 3**0.5, 4**0.5, 5**0.5]
    fig = px.scatter(x=x, y=y)

    visual.print("Square root plot")
    visual.show_figure(fig)
