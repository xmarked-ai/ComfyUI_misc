from nodes import MAX_RESOLUTION
import torch

class ColorCorrectionX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # Black Point
                "black_point_master": ("BOOLEAN", {"default": True, "label_on": "master", "label_off": "channels"}),
                "black_point":   ("FLOAT", {"default": 0.0, "min": -100.0, "step": 0.001}),
                "black_point_r": ("FLOAT", {"default": 0.0, "min": -100.0, "step": 0.001}),
                "black_point_g": ("FLOAT", {"default": 0.0, "min": -100.0, "step": 0.001}),
                "black_point_b": ("FLOAT", {"default": 0.0, "min": -100.0, "step": 0.001}),
                # White Point
                "white_point_master": ("BOOLEAN", {"default": True, "label_on": "master", "label_off": "channels"}),
                "white_point":   ("FLOAT", {"default": 1.0, "min": -100.0, "step": 0.001}),
                "white_point_r": ("FLOAT", {"default": 1.0, "min": -100.0, "step": 0.001}),
                "white_point_g": ("FLOAT", {"default": 1.0, "min": -100.0, "step": 0.001}),
                "white_point_b": ("FLOAT", {"default": 1.0, "min": -100.0, "step": 0.001}),
                # Gain
                "gain_master": ("BOOLEAN", {"default": True, "label_on": "master", "label_off": "channels"}),
                "gain":   ("FLOAT", {"default": 1.0, "step": 0.001}),
                "gain_r": ("FLOAT", {"default": 1.0, "step": 0.001}),
                "gain_g": ("FLOAT", {"default": 1.0, "step": 0.001}),
                "gain_b": ("FLOAT", {"default": 1.0, "step": 0.001}),
                # Multiply
                "multiply_master": ("BOOLEAN", {"default": True, "label_on": "master", "label_off": "channels"}),
                "multiply":   ("FLOAT", {"default": 1.0, "step": 0.001}),
                "multiply_r": ("FLOAT", {"default": 1.0, "step": 0.001}),
                "multiply_g": ("FLOAT", {"default": 1.0, "step": 0.001}),
                "multiply_b": ("FLOAT", {"default": 1.0, "step": 0.001}),
                # Offset
                "offset_master": ("BOOLEAN", {"default": True, "label_on": "master", "label_off": "channels"}),
                "offset":   ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.001}),
                "offset_r": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.001}),
                "offset_g": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.001}),
                "offset_b": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.001}),
                # Gamma
                "gamma_master": ("BOOLEAN", {"default": True, "label_on": "master", "label_off": "channels"}),
                "gamma":   ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.001}),
                "gamma_r": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.001}),
                "gamma_g": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.001}),
                "gamma_b": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.001}),
                # Brightness
                "brightness_master": ("BOOLEAN", {"default": True, "label_on": "master", "label_off": "channels"}),
                "brightness":   ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "brightness_r": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "brightness_g": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "brightness_b": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                # Saturation
                "saturation": ("FLOAT", {"default": 1.0, "min": -100.0, "step": 0.001}),
                # Vibrance
                "vibrance":   ("FLOAT", {"default": 0.0, "step": 0.001}),
                # Clamps
                "black_clamp": ("BOOLEAN", {"default": True, "label_on": "clamp", "label_off": "off"}),
                "white_clamp": ("BOOLEAN", {"default": True, "label_on": "clamp", "label_off": "off"}),
                "invert": (["off", "before", "after"],),
                "mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},),
                "srgb": ("BOOLEAN", {"default": True, "label_on": "sRGB", "label_off": "off"}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "correct_colors"
    CATEGORY = "xmtools/nodes"

    def correct_colors(self, image,
                       black_point, black_point_r, black_point_g, black_point_b, black_point_master,
                       white_point, white_point_r, white_point_g, white_point_b, white_point_master,
                       gain, gain_r, gain_g, gain_b, gain_master,
                       multiply, multiply_r, multiply_g, multiply_b, multiply_master,
                       offset, offset_r, offset_g, offset_b, offset_master,
                       gamma, gamma_r, gamma_g, gamma_b, gamma_master,
                       brightness, brightness_r, brightness_g, brightness_b, brightness_master,
                       saturation, vibrance, black_clamp, white_clamp, mix, srgb, invert, mask=None):
        out = image.clone()  # [batch, height, width, channels]

        if invert == "before":
            out = 1 - out

        if srgb:
            threshold = 0.04045
            out = torch.where(out <= threshold, out / 12.92, ((out + 0.055) / 1.055).pow(2.4))

        # Black Point and White Point
        if black_point_master:
            bp_tensor = torch.tensor([black_point, black_point, black_point])
        else:
            bp_tensor = torch.tensor([black_point_r, black_point_g, black_point_b])

        if white_point_master:
            wp_tensor = torch.tensor([white_point, white_point, white_point])
        else:
            wp_tensor = torch.tensor([white_point_r, white_point_g, white_point_b])

        bp = bp_tensor.view(1, 1, 1, 3)
        wp = wp_tensor.view(1, 1, 1, 3)

        out = (out - bp) / (wp - bp)

        # Gain
        if gain_master:
            out = out * gain
        else:
            out[..., 0] = out[..., 0] * gain_r
            out[..., 1] = out[..., 1] * gain_g
            out[..., 2] = out[..., 2] * gain_b

        # Multiply
        if multiply_master:
            out = out * multiply
        else:
            out[..., 0] = out[..., 0] * multiply_r
            out[..., 1] = out[..., 1] * multiply_g
            out[..., 2] = out[..., 2] * multiply_b

        if offset_master:
            out = out + offset
        else:
            out[..., 0] = out[..., 0] + offset_r
            out[..., 1] = out[..., 1] + offset_g
            out[..., 2] = out[..., 2] + offset_b

        # Gamma
        if gamma_master:
            out = torch.pow(out, 1/gamma)
        else:
            out[..., 0] = torch.pow(out[..., 0], 1/gamma_r)
            out[..., 1] = torch.pow(out[..., 1], 1/gamma_g)
            out[..., 2] = torch.pow(out[..., 2], 1/gamma_b)

        # Brightness
        if brightness_master:
            if brightness >= 0:
                out = 1.0 - (1.0 - out) * (1.0 - brightness)
            else:
                out = out * (1.0 + brightness)
        else:
            for i, br in enumerate([brightness_r, brightness_g, brightness_b]):
                if br >= 0:
                    out[..., i] = 1.0 - (1.0 - out[..., i]) * (1.0 - br)
                else:
                    out[..., i] = out[..., i] * (1.0 + br)

        # Saturation
        r, g, b = out.unbind(dim=-1)
        L = r * 0.299 + g * 0.587 + b * 0.114
        L = L.unsqueeze(-1)
        out = torch.lerp(L, out, saturation)

        # Vibrance
        r, g, b = out.unbind(dim=-1)
        L = r * 0.299 + g * 0.587 + b * 0.114
        L = L.unsqueeze(-1)

        # Оценка насыщенности
        max_rgb, _ = out.max(dim=-1, keepdim=True)
        min_rgb, _ = out.min(dim=-1, keepdim=True)
        saturation_estimate = max_rgb - min_rgb

        # Добавляем порог и сглаживание
        threshold = 0.2  # Порог насыщенности, ниже которого эффект сильнее
        smoothness = 1.0 - torch.tanh((saturation_estimate - threshold) * 5.0)  # Сглаживание перехода
        adjust_factor = vibrance * smoothness * (1.0 - saturation_estimate)

        # Применяем vibrance
        vibrance_weight = 1.0 + adjust_factor
        out = torch.lerp(L, out, vibrance_weight)

        # Black and White clamps
        if black_clamp:
            out = torch.maximum(out, torch.tensor(0))
        if white_clamp:
            out = torch.minimum(out, torch.tensor(1))

        if srgb:
            threshold = 0.0031308
            out = torch.where(out <= threshold, out * 12.92, 1.055 * out.pow(1/2.4) - 0.055)

        if mix < 1.0:
            out = torch.lerp(image, out, mix)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(out)
            out = torch.lerp(image, out, mask_expanded)

        if invert == "after":
            out = 1 - out

        return (out,)


CCX_CLASS_MAPPINGS = {
    "ColorCorrectionX": ColorCorrectionX,
}

CCX_NAME_MAPPINGS = {
    "ColorCorrectionX": "Color Correction X",
}
