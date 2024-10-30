import torch
from torchvision import transforms
import PIL.Image
import numpy as np
import math

class DepthDisplace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "strength": ("FLOAT", {"default": 20.0, "min": 0, "max": 100, "step": 0.1}),
                "blur_depth": ("FLOAT", {"default": 0, "min": 0, "max": 50, "step": 0.1}),
                "offset_x": ("INT", {"default": 0, "min": -2000, "max": 2000, "step": 1}),
                "offset_y": ("INT", {"default": 0, "min": -2000, "max": 2000, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
                "background": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "composed_image", "offset_x", "offset_y")
    FUNCTION = "depthdisplace"

    CATEGORY = "image/filters"

    def depthdisplace(self, image, depth_map, strength, blur_depth, offset_x=0, offset_y=0, mask=None, background=None):
        if image.shape[0] != depth_map.shape[0]:
            raise Exception("Batch size for image and normals must match")

        if image.shape[1] > depth_map.shape[1] or image.shape[2] > depth_map.shape[2]:
            raise Exception("The depth_map dimensions could not be less than image dimensions")

        displaced_image = image.detach().clone()

        if mask != None:
            displaced_mask = mask.detach().clone()

        if background != None:
            composed_image = background.detach().clone()





        return (displaced_image, displaced_mask, composed_image, offset_x, offset_y )

DEPTHDISPLACE_CLASS_MAPPINGS = {
    "DepthDisplace": DepthDisplace,
}

DEPTHDISPLACE_NAME_MAPPINGS = {
    "DepthDisplace": "Depth Displace",
}
