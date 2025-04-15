import torch

MAX_RESOLUTION = 16384

class ConstantColorX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 1}),
                "r": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "g": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "b": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "xmtools/nodes"

    def generate(self, r, g, b, width=1024, height=1024, image=None):
        if image is not None:
            batch, height, width, channels = image.shape
        else:
            batch = 1

        color_tensor = torch.zeros((batch, height, width, 3), dtype=torch.float32, device="cpu")
        color_tensor[:, :, :, 0] = r
        color_tensor[:, :, :, 1] = g
        color_tensor[:, :, :, 2] = b

        return (color_tensor,)

CONSTANTCOLORX_CLASS_MAPPINGS = {
    "ConstantColorX": ConstantColorX,
}

CONSTANTCOLORX_NAME_MAPPINGS = {
    "ConstantColorX": "Constant Color X",
}
