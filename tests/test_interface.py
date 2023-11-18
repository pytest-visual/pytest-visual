from pathlib import Path

from PIL import Image
import numpy as np

from visual.interface import VisualFixture, visual

data_path = Path("examples") / "end_to_end" / "test_data"
image_path = data_path / "train" / "0e171d4bb745_02_08.jpg"

def test_show_image(visual: VisualFixture):
    image = Image.open(image_path)
    visual.print("Show an image:")
    visual.show_image(np.asarray(image))
    
