import random
from pathlib import Path
from shutil import copyfile

def split_data(image_path, mask_path, train_path, test_path, train_ratio, seed):
    """Splits images and corresponding masks into training and testing sets."""
    random.seed(seed)

    image_path = Path(image_path)
    for file in image_path.glob("*.png"):
        name = file.name
        base_name = file.stem  # Without extension

        num = random.random()
        target_folder = Path(train_path) if num < train_ratio else Path(test_path)
        copy_files(base_name, image_path, mask_path, target_folder)

def copy_files(base_name, image_path, mask_path, target_folder):
    """Copies image and all matching masks to the target folder."""
    target_folder = Path(target_folder)
    target_folder.joinpath("images").mkdir(parents=True, exist_ok=True)
    target_folder.joinpath("masks").mkdir(parents=True, exist_ok=True)

    # Copy image
    image_src = Path(image_path) / f"{base_name}.png"
    image_dst = target_folder / "images" / f"{base_name}.png"
    copyfile(image_src, image_dst)

    # Copy all masks that match the image base name (e.g. name_ZP.png)
    mask_path = Path(mask_path)
    for mask_file in mask_path.glob(f"{base_name}_*.png"):
        mask_dst = target_folder / "masks" / mask_file.name
        copyfile(mask_file, mask_dst)

if __name__ == "__main__":
    image_path = r'data\raw\images'
    mask_path = r'data\raw\masks_full_255'
    train_path = r'data\train'
    test_path = r'data\test'
    train_ratio = 0.85
    seed = 22

    split_data(image_path, mask_path, train_path, test_path, train_ratio, seed)
