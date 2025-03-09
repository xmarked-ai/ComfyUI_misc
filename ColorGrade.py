from nodes import MAX_RESOLUTION
import torch

class WhiteBalanceX:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image":       ("IMAGE",),
                "color_red":   ("INT",),
                "color_green": ("INT",),
                "color_blue":  ("INT",),
                "color_space": ("BOOLEAN", {"default": False, "label_on": "RGB", "label_off": "XYZ", "forceInput": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("IMAGE_RGB", "IMAGE_XYZ")
    FUNCTION = "whitebalance"
    CATEGORY = "xmtools/nodes"

    def whitebalance(self, image, color_red, color_green, color_blue, color_space):
        # Conversion matrices RGB to XYZ and XYZ to RGB (CIE 1931 2° observer)
        rgb_to_xyz_matrix = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], dtype=torch.float32)

        xyz_to_rgb_matrix = torch.tensor([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252]
        ], dtype=torch.float32)

        image_RGB = image.detach().clone()

        def rgb_to_xyz(image_rgb):
            # Transform tensor (1, Y, X, 3) to (1, 3, Y, X)
            image_temp = image_rgb.permute(0, 3, 1, 2)

            # Convert to XYZ
            image_xyz = torch.einsum('bchw,cd->bdhw', image_temp, rgb_to_xyz_matrix)

            # Transform tensor back to (1, Y, X, 3)
            image_xyz = image_xyz.permute(0, 2, 3, 1)

            return image_xyz

        def xyz_to_rgb(image_xyz):
            # Transform tensor (1, Y, X, 3) to (1, 3, Y, X)
            image_xyz = image_xyz.permute(0, 3, 1, 2)

            # Convert to RGB
            image_rgb = torch.einsum('bchw,cd->bdhw', image_xyz, xyz_to_rgb_matrix)

            # Transform tensor back to (1, Y, X, 3)
            image_rgb = image_rgb.permute(0, 2, 3, 1)

            return image_rgb

        if color_space:
            image_XYZ = image.detach().clone()

            L = image_RGB[0, :, :, 0] * 0.299 + image_RGB[0, :, :, 1] * 0.587 + image_RGB[0, :, :, 2] * 0.114

            image_RGB[0, :, :, 0] = 1 / (color_red   / 255) * image_RGB[0, :, :, 0]
            image_RGB[0, :, :, 1] = 1 / (color_green / 255) * image_RGB[0, :, :, 1]
            image_RGB[0, :, :, 2] = 1 / (color_blue  / 255) * image_RGB[0, :, :, 2]

            Ln = image_RGB[0, :, :, 0] * 0.299 + image_RGB[0, :, :, 1] * 0.587 + image_RGB[0, :, :, 2] * 0.114

            balanced_image_RGB = image_RGB

            balanced_image_RGB[0, :, :, 0] = image_RGB[0, :, :, 0] / Ln * L
            balanced_image_RGB[0, :, :, 1] = image_RGB[0, :, :, 1] / Ln * L
            balanced_image_RGB[0, :, :, 2] = image_RGB[0, :, :, 2] / Ln * L

        else:
            # red_color, green_color, blue_color to XYZ as sample tensor
            rgb_values = torch.zeros(1, image_RGB.shape[1], image_RGB.shape[2], 3, dtype=torch.float32)
            rgb_values[0, :, :, 0] = color_red   / 255
            rgb_values[0, :, :, 1] = color_green / 255
            rgb_values[0, :, :, 2] = color_blue  / 255
            sample_XYZ = rgb_to_xyz(rgb_values)

            # make Lightness tensor
            rgb_values[0, :, :, 0] = rgb_values[0, :, :, 0] * 0.299 + rgb_values[0, :, :, 1] * 0.587 + rgb_values[0, :, :, 2] * 0.114
            rgb_values[0, :, :, 1] = rgb_values[0, :, :, 0]
            rgb_values[0, :, :, 2] = rgb_values[0, :, :, 0]

            # make sample without lightness in XYZ space
            grey_XYZ = rgb_to_xyz(rgb_values) / sample_XYZ

            # Get lightness of source image before white balance
            L = image_RGB[0, :, :, 0] * 0.299 + image_RGB[0, :, :, 1] * 0.587 + image_RGB[0, :, :, 2] * 0.114

            # Convert and white balance in XYZ space
            image_XYZ = rgb_to_xyz(image_RGB)
            balanced_image_XYZ = grey_XYZ * image_XYZ
            balanced_image_RGB = xyz_to_rgb(balanced_image_XYZ)

            # Get lightness of source image after white balance
            Ln = balanced_image_RGB[0, :, :, 0] * 0.299 + balanced_image_RGB[0, :, :, 1] * 0.587 + balanced_image_RGB[0, :, :, 2] * 0.114

            # Restore lightness usin source image lightness
            balanced_image_RGB[0, :, :, 0] = balanced_image_RGB[0, :, :, 0] / Ln * L
            balanced_image_RGB[0, :, :, 1] = balanced_image_RGB[0, :, :, 1] / Ln * L
            balanced_image_RGB[0, :, :, 2] = balanced_image_RGB[0, :, :, 2] / Ln * L

        return (balanced_image_RGB, image_XYZ)

GRADE_CLASS_MAPPINGS = {
    "WhiteBalanceX": WhiteBalanceX,
}

GRADE_NAME_MAPPINGS = {
    "WhiteBalanceX": "White Balance X",
}
