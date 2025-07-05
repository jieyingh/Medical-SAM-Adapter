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
        name = self.cases[index]
        img_path = os.path.join(self.image_dir, name + '.png')
        image = Image.open(img_path).convert('RGB')

        box = [0, 0, 0, 0]
        pt = (-1, -1)
        p_label = 1  # Default to positive
        
        print(f"Loading image: {name}, Mode: {self.mode}, Prompt: {self.prompt}")

        if self.mode != 'test':
            mask_path = os.path.join(self.mask_dir, name + f'_{self.label}.png')
            mask = Image.open(mask_path).convert('L')
            
            print(f"Loading mask: {mask_path} in mode {self.mode}")

        if self.mode == 'train':
            augmented = self.shared_transform(image=np.array(image), mask=np.array(mask))
            augmented['image'] = self.img_transform(augmented['image'])['image']
            final = ToTensorV2()(image=augmented['image'], mask=augmented['mask'])
            image, mask = final['image'], final['mask']

            print(f"loading image: {name}, Mode: {self.mode}, Prompt: {self.prompt}, Image shape: {image.shape}, Mask shape: {mask.shape}")
            
        elif self.mode == 'val':
            processed = self.infer_transform(image=np.array(image), mask=np.array(mask))
            image = processed['image']
            mask = processed['mask']
            print(f"loading image: {name}, Mode: {self.mode}, Prompt: {self.prompt}, Image shape: {image.shape}, Mask shape: {mask.shape}")

        elif self.mode == 'test':
            image = self.infer_transform(image=np.array(image))['image']

        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'train', 'val', or 'test'.")

        if self.mode != 'test':
            mask = mask.long()

            if self.prompt == 'click':
                p_label, pt = random_click(np.array(mask), point_labels=1)

            elif self.prompt == 'box':
                x_min, x_max, y_min, y_max = random_box(mask.unsqueeze(0).unsqueeze(0).float())
                box = [x_min, x_max, y_min, y_max]

        # printing everything returned
        print(f"Image: {name}, Mode: {self.mode}, Point Label: {p_label}, Point: {pt}, Box: {box}")

        return {
            'image': image,
            'label': mask if self.mode != 'test' else None,
            'p_label': p_label,
            'pt': pt,
            'box': box,
            'image_meta_dict': {'filename_or_obj': name}
        }
