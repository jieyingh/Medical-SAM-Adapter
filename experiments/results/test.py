from PIL import Image
import numpy as np
import os

# Example paths
pred_path = r"experiments\results\01-VanillaPt\predictions\ADAR2_OO.png"
gt_path = r"data\test\masks\ADAR2_OO.png"

# Load images
pred_img = Image.open(pred_path).convert("L")  # grayscale
gt_img = Image.open(gt_path).convert("L")

gt_raw = np.array(gt_img)
if set(np.unique(gt_raw)) <= {0, 1}:
    gt_raw = gt_raw * 255
    gt_img = Image.fromarray(gt_raw.astype(np.uint8))

# Resize GT to match prediction
if gt_img.size != pred_img.size:
    print(f"Resizing GT from {gt_img.size} -> {pred_img.size}")
    gt_img = gt_img.resize(pred_img.size, resample=Image.NEAREST)

# Convert to numpy arrays
pred = np.array(pred_img)
gt = np.array(gt_img)

# Debug raw values
print("Unique values in pred:", np.unique(pred))
print("Unique values in gt:", np.unique(gt))

# Convert to binary 0 and 1
pred_bin = (pred >= 128).astype(np.uint8)
gt_bin = (gt >= 128).astype(np.uint8)

# Debug binarized values
print("Sum pred_bin:", np.sum(pred_bin))
print("Sum gt_bin:", np.sum(gt_bin))

# Now compute Dice
intersection = np.sum(pred_bin * gt_bin)
size_sum = np.sum(pred_bin) + np.sum(gt_bin)
dice = 1.0 if size_sum == 0 else 2 * intersection / size_sum
print("Dice:", dice)
