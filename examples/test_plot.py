from unitvis.visualize import Visualize
import matplotlib.pyplot as plt

def test_plot(visualize: Visualize):
    n = 5
    x, y = [], []
    for i in range(n):
        x.append(i)
        y.append(i**2)
    plt.plot(x, y)

    visualize.print(f"Number of points: {n}")
    visualize.plot(plt)
