import os
import argparse
import numpy as np
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
from scipy.ndimage import binary_erosion, distance_transform_edt

LABEL_ORDER = ["ZP", "PVS", "OO", "PB"]
LABEL_TO_CLASS = {label: i + 1 for i, label in enumerate(LABEL_ORDER)}
TOLERANCE = 5

def dice_coefficient(pred, gt):
    intersection = np.sum(pred * gt)
    size_sum = np.sum(pred) + np.sum(gt)
    if size_sum == 0:
        return 1.0
    return 2.0 * intersection / size_sum

def extract_boundary(mask):
    eroded = binary_erosion(mask)
    return mask ^ eroded

def nsd(gt_mask, pred_mask, tau):
    gt_boundary = extract_boundary(gt_mask)
    pred_boundary = extract_boundary(pred_mask)

    if not np.any(gt_boundary) and not np.any(pred_boundary):
        return 1.0
    if not np.any(gt_boundary) or not np.any(pred_boundary):
        return 0.0

    dist_pred = distance_transform_edt(~pred_boundary)
    dist_gt = distance_transform_edt(~gt_boundary)

    match_gt = (gt_boundary & (dist_pred <= tau))
    match_pred = (pred_boundary & (dist_gt <= tau))

    return (np.sum(match_gt) + np.sum(match_pred)) / (np.sum(gt_boundary) + np.sum(pred_boundary))

def evaluate(gt_folder, pred_folder):
    dice_scores = defaultdict(list)
    nsd_scores = defaultdict(list)

    pred_files = [f for f in os.listdir(pred_folder) if f.endswith(".png")]

    for file in tqdm(pred_files):
        name, label = os.path.splitext(file)[0].rsplit("_", 1)
        if label not in LABEL_TO_CLASS:
            continue

        cls = LABEL_TO_CLASS[label]
        pred_path = os.path.join(pred_folder, file)
        gt_path = os.path.join(gt_folder, file)

        if not os.path.exists(gt_path):
            print(f"Missing GT for {file}, skipping.")
            continue

        # Load images
        pred_img = Image.open(pred_path).convert("L")
        gt_img = Image.open(gt_path).convert("L")

        # Scale GT to 0–255 if needed
        gt_arr = np.array(gt_img)
        if set(np.unique(gt_arr)) <= {0, 1}:
            gt_img = Image.fromarray((gt_arr * 255).astype(np.uint8))

        # Resize GT to match prediction
        if gt_img.size != pred_img.size:
            gt_img = gt_img.resize(pred_img.size, resample=Image.NEAREST)

        # Convert to binary arrays
        pred = (np.array(pred_img) >= 128).astype(np.uint8)
        gt = (np.array(gt_img) >= 128).astype(np.uint8)

        # Compute metrics
        dice = dice_coefficient(pred, gt)
        nsd_score = nsd(gt, pred, TOLERANCE)

        dice_scores[cls].append(dice)
        nsd_scores[cls].append(nsd_score)

    print("\nEvaluation Summary:")
    for label, cls in LABEL_TO_CLASS.items():
        mean_dice = np.mean(dice_scores[cls]) if dice_scores[cls] else 0
        mean_nsd = np.mean(nsd_scores[cls]) if nsd_scores[cls] else 0
        print(f"{label} (Class {cls}): Dice = {mean_dice:.4f}, NSD = {mean_nsd:.4f}")

    macro_dice = np.mean([np.mean(dice_scores[cls]) for cls in LABEL_TO_CLASS.values()])
    macro_nsd = np.mean([np.mean(nsd_scores[cls]) for cls in LABEL_TO_CLASS.values()])
    print(f"\nMacro Dice: {macro_dice:.4f}")
    print(f"Macro NSD: {macro_nsd:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="Folder with ground truth separated masks")
    parser.add_argument("--pred", required=True, help="Folder with predicted separated masks")
    args = parser.parse_args()
    evaluate(args.gt, args.pred)
