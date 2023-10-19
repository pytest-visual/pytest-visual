from unitvis.plugin import Visualize
import plotly.express as px

def test_show_square(visualize: Visualize):
    x, y = [0, 1, 2, 3, 4, 5], [0, 1, 4, 9, 16, 25]
    fig = px.scatter(x=x, y=y)

    visualize.print(f"Square plot")
    visualize.show(fig)

def test_show_square_root(visualize: Visualize):
    x, y = [0, 1, 2, 3, 4, 5], [0, 1**0.5, 2**0.5, 3**0.5, 4**0.5, 5**0.5]
    fig = px.scatter(x=x, y=y)

    visualize.print(f"Square root plot")
    visualize.show(fig)
