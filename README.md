# pytest-visual - an elegant and scalable alternative to one-off visualization scripts

`pytest-visual` introduces the concept of visual testing to computer vision.

## Why visual testing in the computer vision context?

Visualization scripts are extremely useful when developing computer vision applications. However, because running and verifying each script is a manual process, bugs often go unnoticed. With `pytest-visual` however, visualizations are organized as part of unit tests, and you'll be notified of any changes in visualization outputs.

Here's a quick comparison between visual testing and alternative approaches:

| Visualization scripts                                 | Visual testing                                                    |
| ----------------------------------------------------- | ----------------------------------------------------------------- |
| May fail to catch bugs, as changes often go unnoticed | You'll always be notified of any changes in visualization outputs |
| Easily get out of sync with main codebase             | Visualizations are continuously tested to be in sync              |
| Tend to be poorly organized                           | Organization is enforced by the testing framework                 |

As such, visual testing suits much better for ensuring long-term code quality.

## Example

In this example, we demonstrate how to create a basic visual test case. We first create a Python file with a prefix `test_*` (eg. `test_data_augmentation.py`), and add this function:

```python
from visual.interface import VisualFixture, standardize

# `visual` fixture indicates that the test results must be manually verified
# `fix_seeds` fixture ensures deterministic results
def test_show_augmentations(visual, fix_seeds):
    # Create augmented images
    base_image = torchvision.io.read_image("examples/assets/doge.jpg")  # Load the base image
    augmented = [augment(base_image) for _ in range(6)]  # Your augmentation function
    augmented = [image.numpy() for image in augmented]  # pytest-visual accepts only numpy images
    augmented = [standardize(image) for image in augmented]  # Standardize image to uint8 with [0, 255] range and HWC format

    # Show visualization
    visual.images(augmented, caption="Doggos augmented")
```

Then run `pytest --visualize`, and open the url displayed in the terminal window (usually `http://127.0.0.1:54545`). If the visualization looks OK, click "Accept", and pytest should complete with success.

You should see the following output:
![A before and after image showing the effect of data augmentation on a picture of a dog.](examples/screenshots/data_augmentation.jpg?raw=true)

Note that the left side is empty as this is the first time accepting this test case.

The full code together with the provided `augment()` function is available at `examples/test_data_augmentation.py`. See [Installation](#installation) for instructions on how to run these examples.

## Installation

`pytest-visual` can be installed with the following command:

```bash
pip install pytest-visual --upgrade
```

In order to run examples in this repository, you also need development dependencies. You can install them with

```bash
pip install -r requirements-dev.txt
```

For Pytorch model visualization, you'll need to install the graphviz package through your system package management, as described [here](https://graphviz.org/download/). You also have to install torchview (it isn't included by default, as this also installs the rather heavyweight Pytorch, and some people may not need it):

```bash
pip install torchview --upgrade
```

Then go to the root directory of this repository and run visualizations with

```bash
pytest --visual
```

and open your browser at the printed url.

## CLI usage

With `pytest-visual` plugin, the `--visual`, `--visual-accept-all` and `--visual-forget-all` options are added to `pytest`. Here's a quick doc to compare them:

| Command                      | Description                                                                              |
| ---------------------------- | ---------------------------------------------------------------------------------------- |
| `pytest`                     | Run pytest, but skip all tests that are marked as visual.                                |
| `pytest --visual`            | Run pytest with visual tests, and prompt user for manual review on all detected changed. |
| `pytest --visual-accept-all` | Accept all visual tests. Useful when you clone a repository or check out a new branch.   |
| `pytest --visual-forget-all` | Forget all accepted tests.                                                               |

## FAQ

### Can I mix unit testing with visual testing?

Yes. And this is exactly what you should be doing! Just omit the `visual` fixture, and it's a normal `pytest` test again.

### Which parts of a typical computer vision codebase can be visually tested?

Visual testing can be used to verify the correctness of

- data loading and augmentation
- synthetic data generation
- NN model strutures
- loss functions
- learning rate schedules
- detection algorithms
- various classical computer vision algorithms, eg. keypoint detection and matching

### My visual tests are flaky. Why?

This is because your output is not deterministic. For example, plotting a loss over a short training run is probably not a good idea, as it will likely be different for each run. If you need it as a test, consider a normal unit test with some threshold for expected accuracy. Tiny floating point errors could lead to inconsistent output.

That said, we're planning to make `pytest-visual` robust to small fluctuations in output.

### Why even bother testing ML/CV code?

If you've implemented complex data processing logic, custom neural network architectures, advanced post processing steps, or tried to reproduce a paper, you know that bugs can take a very large part of the development effort. Reduced accuracy is often the only symptom of a bug. When conducting an experiment, it's not rare to waste time on

- debugging when your code contains a bug that could have been prevented.
- debugging when your code doesn't contain a bug, but you don't know this.
- revisiting an idea, because you're not sure whether your experiment was correct.
- dropping a good idea because your experiment contained a bug, but you didn't know this.

High quality testing, which includes visualizations, can avoid a large portion of these headaches.

### Isn't automation the whole point of unit testing? Now we have a potentially lazy/irresponsible human in the test loop, who can just blindly "accept" a test.

Full automation is of course great, but even unit tests are written by people. Test quality can vary a lot, and tests written by a lazy/irresponsible person probably won't catch most bugs. Having unit tests won't magically fix your code, but they give a *responsible* developer the tools to write reliable code. Visual testing is one such powerful tool.

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
3. Verify that `pytest-visual` still works end-to-end with `pytest --visual-forget-all && pytest --visual`, and try accepting and declining changes.

Also, please verify that your function inputs and outputs are type-annotated unless there's a good reason not to do so.
