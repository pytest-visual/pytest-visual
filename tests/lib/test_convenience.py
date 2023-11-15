import numpy as np

from visual.lib.convenience import (
    correct_layout,
    get_grid_shape,
    get_image_max_value_from_type,
    get_layout_from_image,
)


def test_get_grid_shape():
    assert get_grid_shape((2, 3), 100) == (2, 3)

    assert get_grid_shape(None, 1) == (1, 1)
    assert get_grid_shape(None, 2) == (1, 2)
    assert get_grid_shape(None, 3) == (2, 2)
    assert get_grid_shape(None, 4) == (2, 2)
    assert get_grid_shape(None, 5) == (2, 3)


def test_get_layout_from_image():
    assert get_layout_from_image("hwc", np.zeros((1, 1, 1))) == "hwc"

    assert get_layout_from_image(None, np.zeros((10, 10, 1))) == "hwc"
    assert get_layout_from_image(None, np.zeros((10, 10, 3))) == "hwc"

    # "hwc", "chw", "hw", "1chw", "1hwc"
    assert get_layout_from_image(None, np.zeros((10, 10, 3))) == "hwc"
    assert get_layout_from_image(None, np.zeros((3, 10, 10))) == "chw"
    assert get_layout_from_image(None, np.zeros((10, 10))) == "hw"
    assert get_layout_from_image(None, np.zeros((1, 3, 10, 10))) == "1chw"
    assert get_layout_from_image(None, np.zeros((1, 10, 10, 3))) == "1hwc"


def test_get_image_max_value_from_type():
    assert get_image_max_value_from_type(1, np.zeros((1,), dtype=np.uint8)) == 1

    assert get_image_max_value_from_type(None, np.zeros((1,), dtype=np.uint8)) == 255
    assert get_image_max_value_from_type(None, np.zeros((1,), dtype=np.float32)) == 1.0
    assert get_image_max_value_from_type(None, np.zeros((1,), dtype=np.float64)) == 1.0


def test_correct_layout():
    image2d = np.zeros((1, 2))
    image3d = np.zeros((1, 2, 3))
    image4d = np.zeros((1, 2, 3, 4))

    assert correct_layout(image2d, "hw").shape == (1, 2)

    assert correct_layout(image3d, "hwc").shape == (1, 2, 3)
    assert correct_layout(image3d, "chw").shape == (2, 3, 1)

    assert correct_layout(image4d, "1hwc").shape == (2, 3, 4)
    assert correct_layout(image4d, "1chw").shape == (3, 4, 2)
