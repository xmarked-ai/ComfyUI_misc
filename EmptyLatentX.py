import torch
import comfy.model_management

class EmptyLatentX:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width":  ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 64}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "flux": ("BOOLEAN", {"default": False, "label_on": "flux", "label_off": "sd15 sdxl", "forceInput": False}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT",)
    RETURN_NAMES = ("LATENT", "width", "height",)

    FUNCTION = "main"
    CATEGORY = "xmtools/nodes"

    def main(self, width, height, image=None, batch_size=1, flux=False):
        if image is not None:
            _, height, width, _  = image.shape
        if flux:
            latent = torch.zeros([batch_size, 16, height // 8, width // 8], device=self.device)
        else:
            latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        return ({"samples":latent}, width, height,)

EMPTYLATENTX_CLASS_MAPPINGS = {
    "EmptyLatentX": EmptyLatentX,
}

EMPTYLATENTX_NAME_MAPPINGS = {
    "EmptyLatentX": "Empty Latent X",
}
