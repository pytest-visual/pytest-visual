# Internals for `pytest-visual`

This document gives a high-level overview of how `pytest-visual` works.

## Overview

Specifically this document describes what happens when you run `pytest --visual`. What is being done when other flags are invoked is not described in this document.

When a user runs `pytest --visual`, the following steps may occur:
* Launch the UI
* Run visual tests
  * Compare currently available visualizations to previously stored visualizations
  * If there's a difference, send visualizations to UI for accept/reject
  * Store the new visualizations if accepted

These steps mostly boil down to UI and persistence.

## UI

TBD.

## Persistence

### Directory structure

Data is stored to disk at `[project-root]/.pytest-visual/`. The directory structure at this location directly corresponds to the directory structure of tests in the project.

Specifically, if the tests are located like this:
```
./tests/test_one.py
./tests/sub/test_two.py
./tests/sub/test_three.py
```

Then the information for each test file will be stored at these directories, respectively:
```
./.pytest-visual/checkpoint/tests/test_one.py/
./.pytest-visual/checkpoint/tests/sub/test_two.py/
./.pytest-visual/checkpoint/tests/sub/test_three.py/
```

Each test case will have its own directory, named after the test case. For example, if `test_one.py` contain test cases `test_something()` and `test_something_else()`, then the visualizations for that test case will be stored at:
```
./.pytest-visual/checkpoint/tests/test_one.py/test_something/
./.pytest-visual/checkpoint/tests/test_one.py/test_something_else/
```

### File structure

Each test case directory contains at least one file, `checkpoint.json`. This file contains the statements run during the test case, in order of execution. If some of the stored statements reference files, those files will be stored in the same directory. The contents of `checkpoint.json` is the following:
```json
{
  "statements": [
    {
      "type": "text" | "images" | "plot",
      "content": "Some text",  # Only for "text" type
      "assets": [] | ["image.1.jpg", "image.2.jpg"] | ["plot.1.png"]
      "hash": "hash of the content"  # Hash of the content, plot or image metadata
      "hash_vectors": array of shape (n_images, hash_vector_size)  # For approximate comparison of "images"
      "metadata": {}  # Additional metadata
    },
  ]
}
```

Assets are stored in the same directory as `checkpoint.json`, and are relative to the directory.

Thus the files for a test case with two images and a plot would look like this:
```
./.pytest-visual/checkpoint/tests/test_one.py/test_something/checkpoint.json
./.pytest-visual/checkpoint/tests/test_one.py/test_something/image.1.jpg
./.pytest-visual/checkpoint/tests/test_one.py/test_something/image.2.jpg
./.pytest-visual/checkpoint/tests/test_one.py/test_something/plot.1.png
```

### Comparison of statements

When a test case is run, the visualizations are compared to the previously stored visualizations. The comparison is done by comparing the hashes of the visualizations. The "hash" field in `checkpoint.json` must match the hash of the current visualization exactly, but the "hash_vectors" field must match up to accuracy specified when calling the function.

All statements must match, or test case is failed.

### `Statement` and `StatementStorage` Pydantic models

In-memory statements are represented by the `Statement` and `StatementStorage` Pydantic models. The `Statement` model contains everything including the assets as numpy arrays or plots, while the `StatementStorage` model only contains string filenames of these assets. Thus `StatementStorage` directly maps one-to-one to the JSON representation, and is directly used during the conversion process. `Statement` on the other hand is used while collecting data from the test case and showing results.

`Statement` and `StatementStorage` can be converted to each other with the `to_storage` and `from_storage` methods.
