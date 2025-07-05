import json
import base64
import io
from PIL import Image, ImageDraw
from pathlib import Path

"""
Converts JSON annotations from DLTA-AI to 8bit mask png images.
Each JSON file contains an image and a list of shapes (polygons) with labels.
"""

def gen_paths(json_file, output_folder):
    """
    Generates image path, mask output path
    """
    base_name = Path(json_file).stem
    image_path = Path(json_file).parent / f"{base_name}.png"
    mask_output_path = Path(output_folder) / f"{base_name}.png"

    return image_path, mask_output_path

def convert_label_to_value(label):
    """
    Converts label to a value for the mask.
    """
    correspondence = {
        "ZP": 1,
        "PVS": 2,
        "OO": 3,
        "PB": 4,
    }

    if label not in correspondence:
        print(f"Warning: Label '{label}' not recognized. Defaulting to 0.")

    return correspondence.get(label, 0)

def json_to_mask(json_file, output_folder):
    
    image_path, mask_output_path = gen_paths(json_file, output_folder)

    # Load JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    # Decode base64 image
    image_data = base64.b64decode(data["imageData"])
    image = Image.open(io.BytesIO(image_data))

    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    # Drawing order
    order = ["ZP", "PVS", "OO", "PB"]

    # Draw each polygon
    for label in order:
        for shape in data["shapes"]:
            if shape["label"] == label:
                points = shape["points"]
                value = convert_label_to_value(label)
                draw.polygon(points, fill=value)
    
    # Save the mask
    mask.save(mask_output_path)

if __name__ == "__main__": 
    annotation_folder = r"data\raw"
    output_folder = r"data\train\masks"

    for json_file in Path(annotation_folder).glob("*.json"):
        json_to_mask(json_file, output_folder)