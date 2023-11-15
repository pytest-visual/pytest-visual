import math
from typing import List, Optional, Tuple

import numpy as np
import plotly.express as px
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots


def get_grid_shape(grid_shape: Optional[Tuple[int, int]], num_images: int) -> Tuple[int, int]:
    """
    Get or calculate the shape of the grid of images to show.
    """
    if grid_shape is not None:
        return grid_shape

    # Try to find a square grid, with the number of rows and columns as close as possible.
    # Prefer more rows than columns.
    cols = math.ceil(num_images**0.5 - 1e-6)
    rows = math.ceil(num_images / cols)
    return rows, cols


def get_layout_from_image(layout: Optional[str], image: np.ndarray) -> str:
    """
    Get or calculate the layout of the grid of images to show.

    Possible values: "hwc", "chw", "hw", "1chw", "1hwc"
    """
    if layout is not None:
        return layout

    matched_layouts = [L for L in ["hwc", "chw", "hw", "1chw", "1hwc"] if layout_matches_image(L, image)]
    assert len(matched_layouts) == 1, f"Could not determine layout from image with shape {image.shape}"
    return matched_layouts[0]


def layout_matches_image(layout: str, image: np.ndarray) -> bool:
    """
    Check if the layout matches the image.
    """
    if len(layout) != len(image.shape):
        return False

    for L, s in zip(layout, image.shape):
        if L == "1" and s != 1:
            return False
        if L == "c" and s not in [1, 3]:
            return False

    return True


def get_image_max_value_from_type(max_value: Optional[float], image: np.ndarray) -> float:
    """
    Get or calculate the maximum value of the image.
    """
    if max_value is not None:
        return max_value

    if image.dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64]:
        return 255.0
    if image.dtype in [np.float16, np.float32, np.float64]:
        return 1.0
    raise ValueError(f"Could not determine max value from image with dtype {image.dtype}")


def create_plot_from_images(
    images: List[np.ndarray],
    grid_shape: Tuple[int, int],
    layout: str,
    mean_denorm: Optional[List[float]],
    std_denorm: Optional[List[float]],
    min_value: float,
    max_value: float,
    height: float,
) -> Figure:
    rows, cols = grid_shape

    # Create multiple augmented images and add them to a grid
    fig = make_subplots(rows=rows, cols=cols, horizontal_spacing=0.05, vertical_spacing=0.05)
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if i < len(images):
                image = images[i]

                # Denormalize, convert to uint8, and correct layout
                if mean_denorm is not None and std_denorm is not None:
                    image = image * std_denorm + mean_denorm
                image = (image - min_value) / (max_value - min_value) * 255
                image = np.clip(image, 0, 255).astype(np.uint8)
                image = correct_layout(image, layout)

                # Add image to grid
                subfig = px.imshow(image, binary_string=True)
                fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
                fig.add_trace(subfig.data[0], row=r + 1, col=c + 1)

    # Remove axes
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c

            if i < len(images):
                fig.update_xaxes(title_text="", showticklabels=False, showgrid=False, row=r + 1, col=c + 1)
                fig.update_yaxes(title_text="", showticklabels=False, showgrid=False, row=r + 1, col=c + 1)

    # Set height
    fig.update_layout(height=height)
    return fig


def correct_layout(image: np.ndarray, layout: str) -> np.ndarray:
    if layout[0] == "1":
        image = np.squeeze(image, axis=0)
        layout = layout[1:]
    if layout[0] == "c":
        image = np.moveaxis(image, 0, -1)
        layout = layout[1:] + "c"
    return image
