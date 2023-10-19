from unitvis.visualize import Visualize
import plotly.express as px

def test_plot(visualize: Visualize):
    x, y = [0, 1, 2, 3, 4, 5], [0, 1, 4, 9, 16, 25]
    fig = px.scatter(x, y)

    visualize.print(f"Number of points: {len(x)}")
    visualize.show(fig)
