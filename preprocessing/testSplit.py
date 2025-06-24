"""
Takes folder path
Splits masks and images into train and test sets
Saves the split data into respective folders
"""

import random
from pathlib import Path
from glob import glob
from shutil import copyfile

def split_data(image_path, mask_path, train_path, test_path, train_ratio, seed):
    """ Splits images and masks into training and testing sets."""
    random.seed(seed)

    for file in Path(image_path).glob("*.png"):
        name = file.name
        num = random.random()
        if num < train_ratio:
            target_folder = Path(train_path)
        else:
            target_folder = Path(test_path)
        copy_files(name, image_path, mask_path, target_folder)
        

def copy_files(name, image_path, mask_path, target_folder):
    """ Copies image and corresponding mask to the target folder."""
    image = Path(image_path) / name
    mask = Path(mask_path) / name

    copy_image = target_folder / "images" /name
    copy_mask = target_folder / "masks" /name

    copyfile(image, copy_image)
    copyfile(mask, copy_mask)

if __name__ == "__main__":
    image_path = r'data\raw\images'
    mask_path = r'data\raw\masks'
    train_path = r'data\train'
    test_path = r'data\test'
    train_ratio = 0.85
    seed = 22

    split_data(image_path, mask_path, train_path, test_path, train_ratio, seed)

