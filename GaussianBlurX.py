import torch
import torch.nn.functional as F
import comfy.model_management as model_management

class GaussianBlurX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 200.0, "step": 0.1},),
                "linear": ("BOOLEAN", {"default": True, "label_on": "on", "label_off": "off"}),
                "mask_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 200.0, "step": 0.1},),
                "linear_mask": ("BOOLEAN", {"default": False, "label_on": "on", "label_off": "off"}),
                "blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},),
            },
            "optional": {
                "mask": ("MASK", ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "apply_fast_blur"
    CATEGORY = "xmtools/nodes"

    def create_gaussian_kernel(self, kernel_size, sigma):
        if kernel_size % 2 == 0:
            kernel_size += 1
        x = torch.arange(kernel_size) - (kernel_size - 1) / 2
        kernel = torch.exp(-0.5 * (x / sigma)**2)
        kernel = kernel / kernel.sum()
        return kernel

    def apply_separable_blur(self, x, kernel):
        kernel_size = kernel.shape[0]
        padding = kernel_size // 2

        x_conv = x.permute(0, 3, 1, 2)

        kernel_h = kernel.view(1, 1, kernel_size, 1)
        kernel_v = kernel.view(1, 1, 1, kernel_size)

        channels = x_conv.shape[1]

        outputs = []
        for c in range(channels):
            channel = x_conv[:, c:c+1]
            padded = F.pad(channel, (padding, padding, 0, 0), mode="replicate")
            temp = F.conv2d(padded, kernel_h.expand(1, 1, kernel_size, 1), padding=(0, 0))
            padded = F.pad(temp, (0, 0, padding, padding), mode="replicate")
            result = F.conv2d(padded, kernel_v.expand(1, 1, 1, kernel_size), padding=(0, 0))
            outputs.append(result)

        final_output = torch.cat(outputs, dim=1)

        final_output = final_output.permute(0, 2, 3, 1)

        return final_output

    def compute_optimal_kernel_size(self, sigma):
        kernel_size = int(2 * torch.ceil(torch.tensor(3 * sigma)) + 1)
        return max(3, kernel_size)

    def apply_fast_blur(self, image, blur, linear, mask_blur, linear_mask, blend, mask=None):
        img_to_blur = image.detach().clone()
        if linear:
            img_to_blur = torch.where(image <= 0.04045, img_to_blur / 12.92, torch.pow((img_to_blur + 0.055) / 1.055, 2.4))

        sigma = blur / 2.0
        kernel_size = self.compute_optimal_kernel_size(sigma)
        padding = kernel_size // 2
        img_to_blur = img_to_blur.permute(0, 3, 1, 2)
        img_to_blur = F.pad(img_to_blur, (padding, padding, padding, padding), mode="replicate")
        img_to_blur = img_to_blur.permute(0, 2, 3, 1)

        kernel = self.create_gaussian_kernel(kernel_size, sigma)
        result = self.apply_separable_blur(img_to_blur, kernel)

        batch, height, width, channels = image.shape
        start_h = padding
        start_w = padding
        end_h = start_h + height
        end_w = start_w + width
        result = result[:, start_h:end_h, start_w:end_w, :]

        if linear:
            result = torch.where(result <= 0.0031308, result * 12.92, 1.055 * torch.pow(result, 1/2.4) - 0.055)

        if mask is not None:
            if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
                raise ValueError(f"Размеры маски {mask.shape[1]}x{mask.shape[2]} не соответствуют размерам изображения {image.shape[1]}x{image.shape[2]}")
                return (result, mask)
            mask_to_blur = mask.detach().clone()
            mask_to_blur = mask_to_blur.unsqueeze(-1)
            if mask_blur != 0:
                if linear_mask:
                    mask_to_blur = torch.where(mask_to_blur <= 0.04045, mask_to_blur / 12.92, torch.pow((mask_to_blur + 0.055) / 1.055, 2.4))
                sigma = mask_blur / 2.0
                kernel_size = self.compute_optimal_kernel_size(sigma)
                padding = kernel_size // 2
                mask_to_blur = mask_to_blur.permute(0, 3, 1, 2)
                mask_to_blur = F.pad(mask_to_blur, (padding, padding, padding, padding), mode="replicate")
                mask_to_blur = mask_to_blur.permute(0, 2, 3, 1)

                kernel = self.create_gaussian_kernel(kernel_size, sigma)
                mask_result = self.apply_separable_blur(mask_to_blur, kernel)

                if linear_mask:
                    mask_result = torch.where(mask_result <= 0.0031308, mask_result * 12.92, 1.055 * torch.pow(mask_result, 1/2.4) - 0.055)

                batch, height, width = mask.shape
                start_h = padding
                start_w = padding
                end_h = start_h + height
                end_w = start_w + width
                mask_to_blur = mask_result[:, start_h:end_h, start_w:end_w, :]

            mask_expanded = mask_to_blur.expand(-1, -1, -1, result.shape[3])
            if blend < 1.0:
                result = image * (1 - mask_expanded * blend) + result * mask_expanded * blend
            else:
                result = image * (1 - mask_expanded) + result * mask_expanded

            return (result, mask_to_blur.squeeze(-1))

        if blend < 1.0:
            result = image * (1.0 - blend) + result * blend

        return (result, mask,)


GAUSSBLUR_CLASS_MAPPINGS = {
    "GaussianBlurX": GaussianBlurX,
}

GAUSSBLUR_NAME_MAPPINGS = {
    "GaussianBlurX": "Gaussian Blur X",
}
