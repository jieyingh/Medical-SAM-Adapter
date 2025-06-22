import os
import json
from glob import glob
from PIL import Image
from sklearn.model_selection import KFold
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class Ooocyte(Dataset):
    def __init__(self, args, data_path, transform=None, transform_mask=None, mode='train'):
        self.data_path = data_path
        self.image_dir = os.path.join(data_path, 'images')
        self.mask_dir = os.path.join(data_path, 'masks')
        self.mode = mode
        self.transform = transform
        self.transform_mask = transform_mask

        self.fold_file = getattr(args, 'fold_file', None)
        self.fold_index = getattr(args, 'fold', 0)
        self.fold_save_path = getattr(args, 'fold_save_path', os.path.join(data_path, 'folds.json'))

        self.img_paths = sorted(glob(os.path.join(self.image_dir, '*.png')))
        self.img_names = [os.path.basename(p) for p in self.img_paths]

        self.folds = self._get_or_create_folds(self.img_names)
        if mode == 'train':
            self.current_files = [f for i in range(5) if i != self.fold_index for f in self.folds[i]]
        else:  # 'val' or 'test'
            self.current_files = self.folds[self.fold_index]

    def _get_or_create_folds(self, filenames):
        if self.fold_file and os.path.exists(self.fold_file):
            with open(self.fold_file, 'r') as f:
                folds = json.load(f)
        elif os.path.exists(self.fold_save_path):
            with open(self.fold_save_path, 'r') as f:
                folds = json.load(f)
        else:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            splits = list(kf.split(filenames))
            folds = {}
            for i, (_, val_idx) in enumerate(splits):
                folds[i] = [filenames[idx] for idx in val_idx]
            with open(self.fold_save_path, 'w') as f:
                json.dump(folds, f)
        return folds

    def __len__(self):
        return len(self.current_files)

    def __getitem__(self, index):
        name = self.current_files[index]
        img_path = os.path.join(self.image_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # mask values: 0–3

        if self.transform:
            image = self.transform(image)
        else:
            image = TF.to_tensor(image)

        if self.transform_mask:
            mask = self.transform_mask(mask)
        else:
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        # Create a multi-channel one-hot mask: shape (4, H, W)
        num_classes = 4
        h, w = mask.shape
        mask_onehot = torch.zeros((num_classes, h, w), dtype=torch.float32)
        for c in range(num_classes):
            mask_onehot[c] = (mask == c).float()

        return {
            'image': image,
            'label': mask_onehot,
            'p_label': 1,
            'pt': torch.tensor([0, 0], dtype=torch.int32),
            'box': [0, 0, 0, 0],
            'image_meta_dict': {'filename_or_obj': name}
        }
