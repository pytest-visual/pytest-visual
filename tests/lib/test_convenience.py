import numpy as np

from visual.lib.convenience import (
    ceil_division,
    correct_layout,
    get_grid_shape,
    get_image_max_value_from_type,
    get_layout_from_image,
    statement_lists_equal,
)
from visual.lib.models import HashVectors_, MaterialStatement, ReferenceStatement


def test_get_grid_shape():
    assert get_grid_shape(1, 3) == (1, 1)
    assert get_grid_shape(2, 3) == (1, 2)
    assert get_grid_shape(3, 3) == (1, 3)
    assert get_grid_shape(4, 3) == (2, 2)
    assert get_grid_shape(5, 3) == (2, 3)
    assert get_grid_shape(6, 3) == (2, 3)
    assert get_grid_shape(7, 3) == (3, 3)

    assert get_grid_shape(10, 3) == (4, 3)


def test_ceil_division():
    assert ceil_division(19, 10) == 2
    assert ceil_division(20, 10) == 2
    assert ceil_division(21, 10) == 3


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


# Statements


def test_statement_lists_equal():
    refs = [
        ReferenceStatement(Type="text", Hash="", Content=""),
        ReferenceStatement(Type="images", Hash=""),
        ReferenceStatement(Type="images", Hash="abc"),
        ReferenceStatement(Type="images", Hash="abc", HashVectors=HashVectors_(Vectors=[[1.0, 2.0, 3.0]], ErrorThreshold=0.1)),
        ReferenceStatement(Type="images", Hash="abc", HashVectors=HashVectors_(Vectors=[[3.0, 4.0, 5.0]], ErrorThreshold=0.1)),
    ]
    mats = [MaterialStatement(**ref.model_dump()) for ref in refs]

    for mat, ref in zip(mats, refs):
        assert statement_lists_equal([mat], [ref])

    assert not statement_lists_equal([mats[0]], [refs[1]])
    assert not statement_lists_equal([mats[1]], [refs[2]])
    assert not statement_lists_equal([mats[2]], [refs[3]])
    assert not statement_lists_equal([mats[3]], [refs[4]])
