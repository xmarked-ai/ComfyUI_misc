import torch
import torch.nn.functional as F
import os
import sys
from PIL import Image, ImageFilter
from transformers import AutoModelForImageSegmentation
import numpy as np
from torchvision.transforms.functional import normalize

from huggingface_hub import snapshot_download

import comfy.model_management
import folder_paths

class RemoveBackgroundX:
    def __init__(self):
        self.device = device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rmbg_models_dir = os.path.join(folder_paths.models_dir, "RMBG")
        self.check_and_download_rmbg()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "auto_resolution": ("BOOLEAN", {"default": True, "label_on": "get from image", "label_off": "use value below", "forceInput": False}),
                "processing_resolution": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "mask_blur": ("FLOAT", {"default": 0.0, "min": 0, "max": 100, "step": 0.1}),
                "mask_feather_type": ("BOOLEAN", {"default": True, "label_on": "gamma", "label_off": "adaptive erode", "forceInput": False}),
                "feather_size": ("FLOAT", {"default": 0, "min": -1.0, "max": 50.0, "step": 0.1}),
                "invert_mask": ("BOOLEAN", {"default": False, "label_on": "invert mask", "label_off": "do not invert", "forceInput": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("IMAGE", "MASK", "masked image (with alpha)", "mask as an image",)
    FUNCTION = "remove_backgroundX"

    CATEGORY = "xmtools/nodes"

    def check_and_download_rmbg(self):
        model_dir = os.path.join(self.rmbg_models_dir, "RMBG-2.0")
        config_path = os.path.join(model_dir, "config.json")

        if not os.path.exists(config_path):
            print(f"File {config_path} not found. Downloading...")

            os.makedirs(model_dir, exist_ok=True)

            snapshot_download(
                repo_id="briaai/RMBG-2.0",
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                revision="main",
                allow_patterns=["*.safetensors", "config.json", "*.py"]
            )

    def calculate_target_size(self, w, h):
        max_side = max(w, h)
        if max_side < 1024:
            return (1024, 1024)
        else:
            target_size = (int(max_side / 64) + 1) * 64
            return (target_size, target_size)

    def rmbg2_0(self, tensor_batch, auto_resolution, processing_resolution, mask_blur, mask_feather_type, feather_size, invert_mask):
        model = AutoModelForImageSegmentation.from_pretrained(
            os.path.join(self.rmbg_models_dir, "RMBG-2.0"),
            trust_remote_code=True,
            local_files_only=True
        )
        model.to(self.device)
        model.eval()

        batch_size, Y, X, channels = tensor_batch.shape
        masks = []

        for i in range(batch_size):
            img_np = (tensor_batch[i].numpy() * 255).astype(np.uint8)

            img_pil = Image.fromarray(img_np)
            original_size = img_pil.size

            if not auto_resolution:
                target_size = (processing_resolution, processing_resolution)
            else:
                target_size = self.calculate_target_size(*original_size)

            img_resized = img_pil.resize(target_size, Image.BICUBIC)

            img_np_resized = np.array(img_resized)
            img_tensor = torch.tensor(img_np_resized, dtype=torch.float32).permute(2, 0, 1)
            img_tensor = img_tensor.unsqueeze(0) / 255.0

            img_tensor = normalize(img_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img_tensor = img_tensor.to(self.device)

            with torch.no_grad():
                result = model(img_tensor)[-1].sigmoid().cpu()

            result = result.squeeze()

            result_np = result.numpy()
            result_np = (result_np * 255).astype(np.uint8)

            mask_pil = Image.fromarray(result_np)
            mask_resized = mask_pil.resize(original_size, Image.LANCZOS)

            if invert_mask:
                mask_resized = Image.eval(mask_resized, lambda x: 255 - x)

            if mask_blur > 0:
                mask_resized = mask_resized.filter(ImageFilter.GaussianBlur(mask_blur))

            mask_np = np.array(mask_resized)
            mask_tensor = torch.tensor(mask_np, dtype=torch.float32) / 255.0

            adjusted_feather = torch.tensor(feather_size + 1, dtype=torch.float32)
            adjusted_feather = torch.where(adjusted_feather == 0, adjusted_feather + 0.00001, adjusted_feather)

            if mask_feather_type:
                mask_tensor = torch.where(mask_tensor < 1, mask_tensor.pow(adjusted_feather), mask_tensor)
            else:
                mask_tensor = torch.where(mask_tensor < 1, mask_tensor - (feather_size * (1 - mask_tensor)), mask_tensor)

            mask_tensor = mask_tensor.clamp(0, 1)

            masks.append(mask_tensor)

        final_tensor = torch.stack(masks, dim=0)

        return final_tensor

    def rmbg1_4(self, tensor_image):
        return tensor_image

    def remove_backgroundX(self, image, auto_resolution, processing_resolution, mask_blur, mask_feather_type, feather_size, invert_mask):
        tensor_input_image = image.detach().clone()

        tensor_output_mask = self.rmbg2_0(image, auto_resolution, processing_resolution, mask_blur, mask_feather_type, feather_size, invert_mask)

        tensor_output_image = torch.cat([tensor_input_image, tensor_output_mask.unsqueeze(-1)], dim=-1)
        mask_as_image = tensor_output_mask.unsqueeze(-1).repeat(1, 1, 1, 3)

        return (tensor_input_image, tensor_output_mask, tensor_output_image, mask_as_image)

RMBG_CLASS_MAPPINGS = {
    "RemoveBackgroundX": RemoveBackgroundX,
}

RMBG_NAME_MAPPINGS = {
    "RemoveBackgroundX": "Remove Background X",
}
