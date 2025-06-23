import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from pathlib import Path

"""
Visualize and verify 8-bit mask images.
Verifies all masks in folder
"""
directory = r"data\train\masks"
output_folder = r"data\train\visMasks"

for mask_path in Path(directory).glob("*.png"):
    mask = np.array(Image.open(mask_path))
    save_path = Path(output_folder) / Path(mask_path).name
    plt.imshow(mask)
    plt.colorbar()
    plt.imsave(save_path, mask)