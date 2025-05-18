import torch
import torch.nn.functional as F
import comfy.model_management as model_management

class GaussianMaskBlurX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", ),
                "blur": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 200.0, "step": 0.1},),
                "linear": ("BOOLEAN", {"default": False, "label_on": "on", "label_off": "off"}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "apply_fast_mask_blur"
    CATEGORY = "xmtools/nodes"

    def create_gaussian_kernel(self, kernel_size, sigma):
        if kernel_size % 2 == 0:
            kernel_size += 1
        x = torch.arange(kernel_size) - (kernel_size - 1) / 2
        kernel = torch.exp(-0.5 * (x / sigma)**2)
        kernel = kernel / kernel.sum()
        return kernel

    def apply_separable_mask_blur(self, x, kernel):
        kernel_size = kernel.shape[0]
        padding = kernel_size // 2

        x_conv = x.unsqueeze(1)

        kernel_h = kernel.view(1, 1, kernel_size, 1)
        kernel_v = kernel.view(1, 1, 1, kernel_size)

        padded = F.pad(x_conv, (padding, padding, 0, 0), mode="replicate")
        temp = F.conv2d(padded, kernel_h, padding=(0, 0))
        padded = F.pad(temp, (0, 0, padding, padding), mode="replicate")
        result = F.conv2d(padded, kernel_v, padding=(0, 0))

        final_output = result.squeeze(1)

        return final_output

    def compute_optimal_kernel_size(self, sigma):
        kernel_size = int(2 * torch.ceil(torch.tensor(3 * sigma)) + 1)
        return max(3, kernel_size)

    def apply_fast_mask_blur(self, mask, blur, linear):
        if blur > 0:
            mask_to_blur = mask.detach().clone()

            if linear:
                mask_to_blur = torch.where(mask_to_blur <= 0.04045, mask_to_blur / 12.92, torch.pow((mask_to_blur + 0.055) / 1.055, 2.4))

            sigma = blur / 2.0
            kernel_size = self.compute_optimal_kernel_size(sigma)
            padding = kernel_size // 2

            mask_padded = F.pad(mask_to_blur.unsqueeze(1), (padding, padding, padding, padding), mode="replicate").squeeze(1)

            kernel = self.create_gaussian_kernel(kernel_size, sigma)
            blurred_mask = self.apply_separable_mask_blur(mask_padded, kernel)

            batch, height, width = mask.shape
            start_h = padding
            start_w = padding
            end_h = start_h + height
            end_w = start_w + width
            result = blurred_mask[:, start_h:end_h, start_w:end_w]

            if linear:
                result = torch.where(result <= 0.0031308, result * 12.92, 1.055 * torch.pow(result, 1/2.4) - 0.055)

            return (result,)

        return(mask,)

NODE_CLASS_MAPPINGS = {
    "GaussianMaskBlurX": GaussianMaskBlurX,
}

NODE_DISPLAY_DISPLAY_NAME_MAPPINGS = {
    "GaussianMaskBlurX": "Gaussian Mask Blur X",
}
