# pytest-visual

`pytest-visual` introduces the concept of visual testing to ML (with automated change detection!).

## Problem statement

Most ML code is poorly tested. This is at least partially because a lot of ML code is very hard to test meaningfully.

For example, we can easily test whether an image data augmentation pipeline produces an image with the correct shape, but does the image also look realistic? And does augmentation produce enough variation? These tests might succeed, but that doesn't mean that your code works.

## Solution - visual testing for ML

`pytest-visual` aims to streamline the organization of visualizations in ML code, enhancing debugging, verification, and code clarity. It automatically detects changes in visualization outputs, presents them in your browser, and prompts for approval. Accepted changes are memorized and not prompted for again. As a result, your visualizations

- **will actually catch bugs**, as changes won't go unnoticed.
- **are always up to date**, as they're run on every `pytest-visual` invocation.
- **won't clutter your production code**, as they're separated into test cases.

## Example

In this example, we demonstrate how to create a basic visual test case. We first create a Python file with a prefix `test_*` (eg. `test_visualization.py`), and add this function:

```python
# `pytest-visual` accepts `plotly` figures
import plotly.express as px

# `test_` prefix makes this a pytest test
# `visual` argument makes this a visual test
def test_show_square(visual):
    # Plot the test data
    x, y = [0, 1, 2, 3, 4, 5], [0, 1, 4, 9, 16, 25]
    fig = px.scatter(x=x, y=y)

    # You can have multiple print() and show() calls in a single case.
    visualize.print("Square plot")
    visualize.show(fig)
```

Then run `pytest --visualize`, and open the url displayed in the terminal window (most likely `http://127.0.0.1:8050`). If the visualization looks OK, click "Accept", and pytest should complete with success.

![A before and after plot displayed side by side.](examples/screenshots/square_plot.png?raw=true "Title")

Here's a sample output from `examples/test_data_augmentation.py`:
![A before and after image showing the effect of data augmentation on a picture of a dog.](examples/screenshots/data_augmentation.jpg?raw=true)

See [Installation](#installation) for instructions on how to run these examples.

## Installation

`pytest-visual` can be installed with the following command:

```bash
pip install pytest-visual --upgrade
```

In order to run examples in this repository, you also need development dependencies. You can install them with

```bash
pip install -r requirements-dev.txt
```

Then go to the root directory of this repository and run visualizations with

```bash
pytest --visual
```

and open your browser at the printed url. Do note that development dependencies are quite heavy (for example Pytorch CPU), but that's just the nature of most DL frameworks.

## CLI usage

With `pytest-visual` plugin, the following options are added to `pytest`:

| Command                     | Description                                                                                        |
| --------------------------- | -------------------------------------------------------------------------------------------------- |
| `pytest --visual`           | Run pytest with visual tests, and prompt user for manual review on all detected changed.           |
| `pytest --visual-yes-all`   | Accept everything without prompting. Useful when you clone a repository or check out a new branch. |
| `pytest --visual-reset-all` | Mark all cases as declined.                                                                        |

## FAQ

### Can I mix unit testing with visual testing?

Yes. And this is exactly what you should be doing! Just omit the `visual` fixture, and it's a normal `pytest` test again.

### Which parts of a typical ML codebase can be visually tested?

Visual testing can be used to verify the correctness of

- data loading and augmentation
- synthetic data generation
- NN model strutures
- loss functions
- learning rate schedules
- detection algorithms
- various filtering algorithms
- etc.

### My visual tests are flaky. Why?

This is typically because your output is undeterministic. For example, plotting a loss over a short training run is probably not a good idea, as it will likely be different for each run. If you need it as a test, consider a normal unit test with some threshold for expected accuracy. Tiny floating point errors could lead to inconsistent output.

That said, we're planning to make `pytest-visual` robust to small fluctuations in output.

### Why even bother testing ML code?

If you've implemented complex data processing logic, custom neural network architectures, advanced post processing steps, or tried to reproduce a paper, you know that bugs can take a very large part of the development effort. Reduced accuracy is often the only symptom of a bug. When conducting an ML experiment, it's typical to waste a lot of time on

- debugging when your ML code contains a bug,
- debugging when your ML code doesn't contain a bug, but you don't know this,
- revisiting an idea, because you're not sure whether your experiment was correct,
- dropping a good idea because your experiment contained a bug, but you don't know this.

Proper testing, including visual testing, can avoid a large portion of these headaches.

### Isn't automation the whole point of unit testing automation? Now we have a potentially lazy/irresponsible human in the loop, who can just blindly "accept" a test.

Full automation is a of course a great, but even the tests that are automatically run are written by people. Test quality can vary a lot, and tests written by a lazy/irresponsible person probably won't catch many bugs. Just having unit tests won't magically fix your code, but they are a very efficient tool to enable *responsible* developers write reliable code.

## Contributing

Your help is greatly needed and appreciated! Given that the library is still in alpha, there will likely be a lot of changes. Thus the main focus is to get the core features right and fix bugs.

We encourage you to contribute by submitting PRs to improve the library. Usually, the PR should link to an existing issue, where the solution idea has been proposed and discussed. In case of simple fixes, where you can clearly see the problem, you can also directly submit a PR.

To get started with development, install all the required dependencies for development:

```bash
pip install -r requirements.txt && pip install -r requirements-dev.txt && pip install -e .
```

The entry point for the plugin is `visual/interface.py`, you should start exploring from that file. The code should be reasonably documented, and is quite short.

Before submitting a PR, please

1. Run non-visual tests with `pytest`
2. Lint your code with `pre-commit run -a`, and ensure no errors persist
3. Verify that `pytest-visual` still works end-to-end with `pytest --visual-reset-all && pytest --visual`, and try accepting and declining changes.

Also, please verify that your function inputs and outputs are type-annotated unless there's a good reason not to do so.
