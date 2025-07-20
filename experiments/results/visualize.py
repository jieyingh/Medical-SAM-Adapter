import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

# Define label order and associated colors
LABEL_ORDER = ["ZP", "PVS", "OO", "PB"]
CLASS_COLORS = [
    (15, 84, 160),       # Background (class 0)
    (238, 160, 197),     # ZP - pink
    (250, 202, 91),     # PVS - orange
    (255, 246, 144),     # OO - yellow
    (254, 253, 253),   # PB - white
]

def create_combined_mask(name, mask_folder, image_shape):
    """Combines separated binary masks into one multiclass mask."""
    combined_mask = np.zeros(image_shape, dtype=np.uint8)

    for class_idx, label in enumerate(LABEL_ORDER, start=1):
        mask_path = os.path.join(mask_folder, f"{name}_{label}.png")
        if not os.path.exists(mask_path):
            continue

        mask_img = Image.open(mask_path).convert("L")
        mask_np = np.array(mask_img)

        # 🔧 Scale from 0/1 → 0/255 before resizing
        if np.array_equal(np.unique(mask_np), [0, 1]):
            mask_np = mask_np * 255

        # Then resize
        mask_resized = Image.fromarray(mask_np).resize(image_shape[::-1], resample=Image.NEAREST)
        binary_mask = np.array(mask_resized) >= 128

        combined_mask[binary_mask] = class_idx

    return combined_mask

def colorize_mask(mask):
    """Converts multiclass mask to RGB image using CLASS_COLORS."""
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, color in enumerate(CLASS_COLORS):
        color_mask[mask == i] = color
    return color_mask

def overlay(image, mask_color, alpha=0.5):
    """Blends the color mask over the original image."""
    return np.clip(image * (1 - alpha) + mask_color * alpha, 0, 255).astype(np.uint8)

def visualize(mask_folder, image_folder, output_folder):
    """Main visualization routine."""
    os.makedirs(output_folder, exist_ok=True)
    names = sorted(set(f.rsplit("_", 1)[0] for f in os.listdir(mask_folder) if f.endswith(".png")))

    for name in tqdm(names, desc="Visualizing"):
        image_path = os.path.join(image_folder, f"{name}.png")
        if not os.path.exists(image_path):
            print(f"Image missing for {name}, skipping.")
            continue

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Create combined and colorized mask
        mask_np = create_combined_mask(name, mask_folder, image_np.shape[:2])
        mask_color = colorize_mask(mask_np)

        # Overlay on original image
        overlay_img = overlay(image_np, mask_color, alpha=0.5)

        # Save combined mask and overlay
        Image.fromarray(mask_color).save(os.path.join(output_folder, f"{name}_combined.png"))
        Image.fromarray(overlay_img).save(os.path.join(output_folder, f"{name}_overlay.png"))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize separated binary masks.")
    parser.add_argument("--masks", required=True, help="Folder with separated binary masks")
    parser.add_argument("--images", required=True, help="Folder with original .png images")
    parser.add_argument("--output", required=True, help="Folder to save visualizations")
    args = parser.parse_args()

    visualize(args.masks, args.images, args.output)
