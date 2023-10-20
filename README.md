# Unitvis - ML visualization organization and automation framework

## Problem statement

In ML code, visualizations are not only great for exploring the data or verifying the model performance, but also for debugging and verifying ML code. However, in pracice, these visualizations are poorly organized, and lack standardization. Such code is often spread across script files, separate functions, and sometimes just interleaved with training/inference code. This results in several problems:

- **Outdated code.** Because visualizations are usually only run when debugging something, they can easily get out of sync with the main codebase. When you finally try to use them, they often won't work.
- **Visualization won't catch bugs.** Because interpreting visualizations is a manual process, it's often skipped.
- **Visualizations are often only run on end-to-end code.** Lack of isolation makes debugging more time-consuming.
- **Worse code clarity.** Visualizations are often interleaved with training/production code (perhaps within a comment block).

## Solution - unit visualization

**Unit visualization** is to visualization as **unit testing** is to running assert statements.

You organize your visualizations into "visualization cases", just as you might have test cases. These cases are run every time you run `pytest --visualize`. The program asks for your approval on each case that has changed since the last invocation, or which you haven't yet approved. This way, you're not cluttered with unnecessary information, yet always get notified of a change in important code. This results in:

- **Up to date code.** Every visualization is run every time you run `pytest --visualize`
- **Visualization will catch bugs.** You only need to verify those cases that have changed.
- **Visualizations are run on small portions of code.** Use similar logic as you would use with unit tests.
- **Better code clarity.** No interleaving debug/visualization code with production code.

`unitvis` is a `pytest` plugin, so it's usage is very familiar to those who have used `pytest` before.

## Example

Example:

```python
from unitvis import Visualize
import matplotlib.pyplot as plt

def test_plot(visualize: Visualize):
    n = 5
    x, y = []
    for i in range(n):
        x.append(i)
        y.append(i**2)
    plt.plot(x, y)

    visualize.print(f"Number of points: {n}")
    visualize.plot(plt)
```

Note that `visualize` test cases are skipped by default on `pytest`, but can be run with `pytest --visualize`.

The output is only printed, plotted, and asked for manual review, if something changes.

## CLI usage

With unitvis plugin, the following options are added to `pytest`:

- `pytest --visualize`
  - Run pytest with visualization cases, and prompt user for manual review for all changed cases.
- `pytest --visualize-yes-all`
  - Everything is accepted without prompting.
- `pytest --visualize-reset-all`
  - Don't visualize, but mark all cases as unaccepted.
