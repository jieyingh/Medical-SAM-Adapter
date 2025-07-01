import os
from glob import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from albumentations.pytorch import ToTensorV2

"""
Dataset class for Oocyte segmentation task.
This class handles loading images and masks, applying transformations,
and preparing data for training, validation, or testing.
It supports three modes: 'train', 'val', and 'test'.
"""

class Oocyte(Dataset):
    def __init__(self, args, data_path, shared_transform, img_transform, infer_transform, mode='none', prompt='none'):
        self.data_path = data_path
        self.image_dir = os.path.join(data_path, 'images')
        self.mask_dir = os.path.join(data_path, 'masks')
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.cases = sorted([p.stem for p in Path(self.image_dir).glob('*.png')])

        self.shared_transform = shared_transform
        self.img_transform = img_transform
        self.infer_transform = infer_transform

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        point_label = 1 # always positive prompt

        """Get the images"""
        if index >= len(self.cases):
            raise IndexError(f"Index {index} out of bounds for dataset with length {len(self.cases)}")
        
        name = self.cases[index]
        img_path = os.path.join(self.image_dir, name + '.png')
        image = Image.open(img_path).convert('RGB')

        if self.mode != 'test':
            # Load the mask only if not in test mode
            mask_path = os.path.join(self.mask_dir, name + '.png')
            mask = Image.open(mask_path).convert('L')

        """Transform the image and mask based on the mode"""
        if self.mode == 'train':
            augmented = self.shared_transform(image=np.array(image), mask=np.array(mask))
            augmented['image'] = self.img_transform(augmented['image'])['image']
            final = ToTensorV2()(image=augmented['image'], mask=augmented['mask'])
            image, mask = final['image'], final['mask']
            
        elif self.mode == 'val':
            processed = self.infer_transform(image=np.array(image), mask=np.array(mask))
            image = processed['image']
            mask = processed['mask']

        elif self.mode == 'test':
            image = self.infer_transform(image=np.array(image))['image']

        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'train', 'val', or 'test'.")
        
        if self.mode != 'test':
            mask = mask.long()

        return {
            'image': image,
            'label': mask, # the target masks, just need height and width
            'p_label': 1, # prompt label to decide positive/negative prompt. can put 1 if don't need negative prompt
            'pt': torch.tensor([0, 0], dtype=torch.int32), # the prompt. 
            'box': [0, 0, 0, 0],
            'image_meta_dict': {'filename_or_obj': name} 
        }
