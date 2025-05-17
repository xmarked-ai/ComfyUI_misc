import torch
import numpy as np
from PIL import Image

class ImageResizeX:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "size": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 1}),
                "side": (["short", "long"], {"default": "short"}),
                "interpolation": (["nearest-neighbor", "bilinear", "bicubic", "lanczos"], {"default": "lanczos"}),
                "multiple_of": ("INT", {"default": 64, "min": 1, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_image"
    CATEGORY = "xmtools/nodes"

    def resize_image(self, image: torch.Tensor, size: int, side: str, interpolation: str, multiple_of: int):
        batch_size, original_h, original_w, _ = image.shape

        output_images = []

        pil_interpolation_map = {
            "nearest-neighbor": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
        }

        resampling_method = pil_interpolation_map.get(interpolation, Image.Resampling.LANCZOS)

        for i in range(batch_size):
            img_tensor_single = image[i]
            img_pil = Image.fromarray((img_tensor_single.cpu().numpy() * 255).astype(np.uint8))

            pil_w, pil_h = img_pil.size

            if pil_w == 0 or pil_h == 0:
                intermediate_resized_pil = img_pil.copy()
            else:
                if side == "short":
                    if pil_w < pil_h:
                        new_w = size
                        new_h = int(round(pil_h * (size / pil_w))) if pil_w > 0 else 0
                    else:
                        new_h = size
                        new_w = int(round(pil_w * (size / pil_h))) if pil_h > 0 else 0
                elif side == "long":
                    if pil_w > pil_h:
                        new_w = size
                        new_h = int(round(pil_h * (size / pil_w))) if pil_w > 0 else 0
                    else:
                        new_h = size
                        new_w = int(round(pil_w * (size / pil_h))) if pil_h > 0 else 0
                else:
                    new_w, new_h = pil_w, pil_h

                new_w = max(1, new_w)
                new_h = max(1, new_h)
                intermediate_resized_pil = img_pil.resize((new_w, new_h), resample=resampling_method)

            current_w, current_h = intermediate_resized_pil.size
            final_pil_image = intermediate_resized_pil

            if multiple_of > 1 and (current_w > 0 and current_h > 0):
                target_w_mult = (current_w // multiple_of) * multiple_of
                target_h_mult = (current_h // multiple_of) * multiple_of

                if target_w_mult < current_w or target_h_mult < current_h:
                    if target_w_mult == 0 or target_h_mult == 0:
                        final_pil_image = Image.new(intermediate_resized_pil.mode, (target_w_mult, target_h_mult))
                    else:
                        crop_x_start = (current_w - target_w_mult) // 2
                        crop_y_start = (current_h - target_h_mult) // 2
                        crop_box = (crop_x_start, crop_y_start, crop_x_start + target_w_mult, crop_y_start + target_h_mult)
                        final_pil_image = intermediate_resized_pil.crop(crop_box)

            if final_pil_image.width == 0 or final_pil_image.height == 0:
                 num_channels = image.shape[-1]
                 resized_np = np.zeros((final_pil_image.height, final_pil_image.width, num_channels), dtype=np.float32)
            else:
                resized_np = np.array(final_pil_image).astype(np.float32) / 255.0

            resized_tensor = torch.from_numpy(resized_np).unsqueeze(0)
            output_images.append(resized_tensor)

        if not output_images:
             return (image,)

        return (torch.cat(output_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "ImageResizeX": ImageResizeX
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageResizeX": "Image Resize X"
}
