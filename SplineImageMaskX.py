# just a base for developent
import os
import torch
import numpy as np
import json
from PIL import Image, ImageDraw, ImageOps, ImageSequence
import folder_paths
from comfy.utils import common_upscale
import io
import base64
import comfy.utils
import cv2

class SplineImageMask:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files),),
                "points_store": ("STRING", {"multiline": False}),
                "coordinates": ("STRING", {"multiline": False}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "create_mask"
    CATEGORY = "xmtools/nodes"
    DESCRIPTION = """
    Creates a mask from a closed spline drawn on the image.

    ## How to use
    1. Select an image from the list
    2. Use the editor to draw a closed spline on the image
    3. The output is a mask where the area inside the spline is white

    **Shift + click** to add control point at end.
    **Ctrl + click** to add control point between two points.
    **Right click on a point** to delete it.
    """

    def create_mask(self, image, points_store, coordinates):
        # Load the selected image
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)

        if img.mode == 'I':
            img = img.point(lambda i: i * (1 / 255))

        # Convert to RGB if needed
        img = img.convert("RGB")

        # Get image dimensions
        width, height = img.size

        # Parse the coordinates
        try:
            coordinates = json.loads(coordinates.replace("'", '"'))
        except:
            coordinates = []

        # Create a blank mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Extract coordinates as (x, y) tuples for the polygon
        polygon_points = [(int(point['x']), int(point['y'])) for point in coordinates]
        corners_np = np.array([polygon_points], dtype=np.int32)

        # If we have at least 3 points, we can draw a polygon
        if len(polygon_points) >= 3:
            # Draw the filled polygon
            cv2.fillPoly(mask, corners_np, color=255, lineType=cv2.LINE_AA)

        # Convert to tensor
        mask_tensor = torch.from_numpy(mask).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension

        # Create response with background image for JS
        pil_image = img.copy()
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=75)
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        return (mask_tensor,)

# Register nodes
NODE_CLASS_MAPPINGS = {
    "SplineImageMask": SplineImageMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SplineImageMask": "Spline Image Mask X",
}
