from pathlib import Path

import numpy as np
from PIL import Image

from visual.interface import VisualFixture, visual

data_path = Path("examples") / "end_to_end" / "test_data"
image_path = data_path / "train" / "0e171d4bb745_02_08.jpg"


def test_show_image(visual: VisualFixture):
    image = Image.open(image_path)
    visual.text("Show an image:")
    visual.image(np.asarray(image))
