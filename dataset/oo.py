import os
from glob import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from albumentations.pytorch import ToTensorV2

from utils import random_click, random_box

"""
Dataset class for Oocyte segmentation task.
This class handles loading images and masks, applying transformations,
and preparing data for training, validation, or testing.
It supports three modes: 'train', 'val', and 'test'.
"""

class Oocyte(Dataset):
    def __init__(self, args, data_path, shared_transform, img_transform, infer_transform, mode='none', prompt='none'):
        self.data_path = data_path
        self.image_dir = os.path.join(data_path, 'images') # implies there is a folder named 'images' in data_path
        self.mask_dir = os.path.join(data_path, 'masks') # implies there is a folder named 'masks' in data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.cases = sorted([p.stem for p in Path(self.image_dir).glob('*.png')])
        self.label = args.label

        self.shared_transform = shared_transform
        self.img_transform = img_transform
        self.infer_transform = infer_transform

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        point_label = 1

        """Get the images"""
        name = self.cases[index]
        img_path = os.path.join(self.image_dir, name + '.png')
        image = Image.open(img_path).convert('RGB')
        # mask_path = os.path.join(self.mask_dir, name + f'_{self.label}.png')
        mask_path = os.path.join(self.mask_dir, name + '.png')  # Assuming masks are named like images
        mask = Image.open(mask_path).convert('L')

        augmented = self.shared_transform(image=np.array(image), mask=np.array(mask))
        augmented['image'] = self.img_transform(image=augmented['image'])['image']

        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask), point_label)

        # else:
        #     pt = np.array([0, 0], dtype=np.int32)

        if self.prompt == 'box':
            x_min, x_max, y_min, y_max = random_box(mask)
            box = [x_min, x_max, y_min, y_max]
        else:
            box = [0, 0, 0, 0]

        final = ToTensorV2()(image=augmented['image'], mask=augmented['mask'])
        image, mask = final['image'], final['mask']

        # printing everything returned
        print(f"Image: {name}, Mode: {self.mode}, Point Label: {point_label}, Point: {pt}, Box: {box}")

        return {
            'image': image,
            'label': mask.unsqueeze(0),
            'p_label': point_label,
            'pt': pt,
            'box': box,
            'image_meta_dict': {'filename_or_obj': name}
        }
