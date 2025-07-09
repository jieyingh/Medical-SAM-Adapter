import json
import base64
import io
from PIL import Image, ImageDraw
from pathlib import Path

"""
Converts JSON annotations from DLTA-AI to 8-bit mask PNG images.
Each label will be converted to a one-hot encoded mask.
1 channel per image.
"""

def gen_paths(json_file, output_folder):
    base_name = Path(json_file).stem
    image_path = Path(json_file).parent / f"{base_name}.png"
    mask_output_path = Path(output_folder) / f"{base_name}"
    return image_path, mask_output_path

def json_to_mask(json_file, output_folder):
    image_path, mask_output_path = gen_paths(json_file, output_folder)

    with open(json_file, "r") as f:
        data = json.load(f)

    image_data = base64.b64decode(data["imageData"])
    image = Image.open(io.BytesIO(image_data))
    img_size = image.size

    # Drawing order and nesting
    order = ["ZP", "PVS", "OO", "PB"]
    inner_map = {"ZP": "PVS", "PVS": "OO"}

    for label in order:
        mask = Image.new("L", img_size, 0)
        draw = ImageDraw.Draw(mask)

        # Draw outer shape(s)
        for shape in data["shapes"]:
            if shape["label"] == label:
                draw.polygon(shape["points"], fill=1)

        # Subtract inner shapes if applicable
        inner_label = inner_map.get(label)
        if inner_label:
            for shape in data["shapes"]:
                if shape["label"] == inner_label:
                    draw.polygon(shape["points"], fill=0)

        mask.save(str(mask_output_path) + f"_{label}.png")

if __name__ == "__main__": 
    annotation_folder = r"data\raw\annotations"
    output_folder = r"data\raw\masks"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for json_file in Path(annotation_folder).glob("*.json"):
        json_to_mask(json_file, output_folder)
