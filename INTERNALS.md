# Internals for `pytest-visual`

This document gives a high-level overview of how `pytest-visual` works.

## Overview

Specifically this document describes what happens when you run `pytest --visual`. What is being done when other flags are invoked is not described in this document.

When a user runs `pytest --visual`, the following steps may occur:

- Launch the UI
- Run visual tests
  - Compare currently available visualizations to previously stored visualizations
  - If there's a difference, send visualizations to UI for accept/reject
  - Store the new visualizations if accepted

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
      "Image": "assets/0.jpg",
      "Caption": "Some text",
      "HashVector": {
        "Vector": array of shape (hash_vector_size),  # For approximate comparison of images
        "ErrorThreshold": float  # For approximate comparison of images
      },
    },
  ]
}
```

Assets are stored in the same directory as `checkpoint.json`, and are relative to the directory.

Thus the files for a test case with two images and a figure would look like this:

```
./.pytest-visual/checkpoint/tests/test_one.py/test_something/checkpoint.json
./.pytest-visual/checkpoint/tests/test_one.py/test_something/assets/0.jpg
./.pytest-visual/checkpoint/tests/test_one.py/test_something/assets/1.jpg
```

### Comparison of statements

When a test case is run, the visualizations are compared to the previously stored visualizations. The comparison is done by comparing the hashes of the visualizations. The `HashVectors` field in `checkpoint.json` must match the hash_vector of the current visualization up to the accuracy specified in `ErrorThreshold` when calling the function.

All statements must match, or test case is failed.

### `Statement` and `OnDiskStatement` Pydantic models

In-memory statements are represented by the `Statement` and `OnDiskStatement` Pydantic models. The `Statement` model contains everything including the asset as numpy array or plotly figure (if applicable), while the `OnDiskStatement` model only contains string filename of the asset (if applicable). Thus `OnDiskStatement` directly maps one-to-one to the JSON representation, and is directly used during the conversion process. `Statement` on the other hand is used while collecting data from the test case and showing results.
