import torch
import torch.nn.functional as F
import numpy as np

def rgb2lab(rgb):
    rgb_lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))

    r, g, b = rgb_lin[..., 0], rgb_lin[..., 1], rgb_lin[..., 2]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    x_n, y_n, z_n = 0.95047, 1.0, 1.08883

    x = x / x_n
    y = y / y_n
    z = z / z_n

    epsilon = 0.008856
    kappa = 903.3

    fx = torch.where(x > epsilon, x.pow(1.0/3.0), (kappa * x + 16.0) / 116.0)
    fy = torch.where(y > epsilon, y.pow(1.0/3.0), (kappa * y + 16.0) / 116.0)
    fz = torch.where(z > epsilon, z.pow(1.0/3.0), (kappa * z + 16.0) / 116.0)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    L = L / 100.0
    a = a / 127.0
    b = b / 127.0

    lab = torch.stack([L, a, b], dim=-1)
    return lab

def lab2rgb(lab):
    L = lab[..., 0] * 100.0
    a = lab[..., 1] * 127.0
    b = lab[..., 2] * 127.0

    x_n, y_n, z_n = 0.95047, 1.0, 1.08883

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    delta = 6.0 / 29.0
    delta_cube = delta * delta * delta

    x = torch.where(
        fx > delta,
        fx.pow(3),
        (fx - 16.0 / 116.0) * 3 * delta * delta
    ) * x_n

    y = torch.where(
        fy > delta,
        fy.pow(3),
        (fy - 16.0 / 116.0) * 3 * delta * delta
    ) * y_n

    z = torch.where(
        fz > delta,
        fz.pow(3),
        (fz - 16.0 / 116.0) * 3 * delta * delta
    ) * z_n

    r = x *  3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y *  1.8760108 + z *  0.0415560
    b = x *  0.0556434 + y * -0.2040259 + z *  1.0572252

    r = torch.where(
        r <= 0.0031308,
        12.92 * r,
        1.055 * r.pow(1.0 / 2.4) - 0.055
    )

    g = torch.where(
        g <= 0.0031308,
        12.92 * g,
        1.055 * g.pow(1.0 / 2.4) - 0.055
    )

    b = torch.where(
        b <= 0.0031308,
        12.92 * b,
        1.055 * b.pow(1.0 / 2.4) - 0.055
    )

    rgb = torch.stack([r, g, b], dim=-1)
    return rgb

def srgb_to_rec709(image):
    threshold = 0.04045
    linear = torch.where(
        image <= threshold,
        image / 12.92,
        ((image + 0.055) / 1.055).pow(2.4)
    )

    transform = torch.tensor([
        [0.2126,  0.7152,  0.0722],
        [-0.1146, -0.3854, 0.5000],
        [0.5000, -0.4542, -0.0458],
    ], device=image.device, dtype=image.dtype)

    ycbcr = torch.tensordot(linear, transform.T, dims=1)
    return ycbcr

def rec709_to_srgb(ycbcr):
    inverse_transform = torch.tensor([
        [1.0,     0.0,      1.5748],
        [1.0,    -0.1873,  -0.4681],
        [1.0,     1.8556,   0.0   ],
    ], device=ycbcr.device, dtype=ycbcr.dtype)

    linear = torch.tensordot(ycbcr, inverse_transform.T, dims=1)

    threshold = 0.0031308
    srgb = torch.where(
        linear <= threshold,
        linear * 12.92,
        1.055 * linear.pow(1/2.4) - 0.055
    )

    return srgb


class ImageCompositionX:
    """Node for compositing two images with optional masks using various blend modes"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "operation": (["atop", "average", "color", "color-burn", "color-dodge", "conjoint-over",
                              "copy", "difference", "disjoint-over", "divide", "exclusion",
                              "from", "geometric", "hard-light", "hypot", "in", "mask",
                              "matte", "max", "min", "minus", "multiply", "out",
                              "over", "overlay", "plus", "screen", "soft-light", "spectral-multiply",
                              "spectral-multiply(rec709)", "stencil", "under", "xor"],),
                "offset_x": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "offset_y": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},),
                "invert_mask_a": ("BOOLEAN", {"default": False}),
                "invert_mask_b": ("BOOLEAN", {"default": False}),
                "premult_a": ("BOOLEAN", {"default": False}),
                "premult_b": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask_a": ("MASK",),
                "mask_b": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask_a", "mask_b", "result_mask")
    FUNCTION = "composite_images"
    CATEGORY = "xmtools/nodes"

    def composite_images(self, image_a, image_b, operation, offset_x=0, offset_y=0, mask_a=None, mask_b=None, invert_mask_a=False, invert_mask_b=False, premult_a=False, premult_b=False, mix=1 ):
        # Make sure we're working with normalized images (0-1 range)
        # Convert to float if necessary
        image_a = image_a.float() if image_a.dtype != torch.float32 else image_a
        image_b = image_b.float() if image_b.dtype != torch.float32 else image_b

        mix_in_ops = False

        # Проверка размеров маски A
        if mask_a is not None:
            # Убедимся, что у маски правильная размерность
            if mask_a.dim() == 2:
                mask_a = mask_a.unsqueeze(0)  # Добавляем размерность батча если отсутствует

            # Проверяем совпадение размеров маски и изображения
            if mask_a.shape[1:3] != image_a.shape[1:3]:
                raise ValueError(f"Размеры маски A ({mask_a.shape[1:3]}) не совпадают с размерами изображения A ({image_a.shape[1:3]})")

            # Применяем инвертирование маски A если нужно
            if invert_mask_a:
                mask_a = 1.0 - mask_a

        # Проверка размеров маски B
        if mask_b is not None:
            # Убедимся, что у маски правильная размерность
            if mask_b.dim() == 2:
                mask_b = mask_b.unsqueeze(0)  # Добавляем размерность батча если отсутствует

            # Проверяем совпадение размеров маски и изображения
            if mask_b.shape[1:3] != image_b.shape[1:3]:
                raise ValueError(f"Размеры маски B ({mask_b.shape[1:3]}) не совпадают с размерами изображения B ({image_b.shape[1:3]})")

            # Применяем инвертирование маски B если нужно
            if invert_mask_b:
                mask_b = 1.0 - mask_b

        # Get dimensions of both images
        batch_a, height_a, width_a, channels_a = image_a.shape
        batch_b, height_b, width_b, channels_b = image_b.shape

        # Create a buffer the size of image_b
        buffer_a = torch.zeros((batch_a, height_b, width_b, channels_a), device=image_a.device)

        # Calculate offsets to center image_a in buffer, then add user offset
        center_offset_y = (height_b - height_a) // 2
        center_offset_x = (width_b - width_a) // 2

        # Apply user offset
        final_offset_y = center_offset_y + offset_y
        final_offset_x = center_offset_x + offset_x

        # Calculate source and destination regions for copying
        # Source region (from image_a)
        src_y_start = max(0, -final_offset_y)
        src_x_start = max(0, -final_offset_x)
        src_y_end = min(height_a, height_b - final_offset_y)
        src_x_end = min(width_a, width_b - final_offset_x)

        # Destination region (in buffer)
        dst_y_start = max(0, final_offset_y)
        dst_x_start = max(0, final_offset_x)
        dst_y_end = min(height_b, final_offset_y + height_a)
        dst_x_end = min(width_b, final_offset_x + width_a)

        # Calculate heights and widths to copy
        copy_height = min(src_y_end - src_y_start, dst_y_end - dst_y_start)
        copy_width = min(src_x_end - src_x_start, dst_x_end - dst_x_start)

        # Only copy if regions overlap
        if copy_height > 0 and copy_width > 0:
            buffer_a[:,
                     dst_y_start:dst_y_start + copy_height,
                     dst_x_start:dst_x_start + copy_width,
                     :] = image_a[:,
                                  src_y_start:src_y_start + copy_height,
                                  src_x_start:src_x_start + copy_width,
                                  :]

        # Create default masks if not provided (all ones)
        if mask_a is None:
            mask_a = torch.ones((image_a.shape[0], image_a.shape[1], image_a.shape[2]),
                               device=image_a.device)
        else:
            # Ensure mask has right dimensions
            if mask_a.dim() == 2:
                mask_a = mask_a.unsqueeze(0)  # Add batch dimension if missing

            # Check if mask_a size matches image_a
            if mask_a.shape[1:3] != image_a.shape[1:3]:
                raise ValueError(f"Mask A dimensions ({mask_a.shape[1:3]}) do not match Image A dimensions ({image_a.shape[1:3]})")

        if mask_b is None:
            mask_b = torch.ones((image_b.shape[0], image_b.shape[1], image_b.shape[2]),
                               device=image_b.device)
        else:
            # Ensure mask has right dimensions
            if mask_b.dim() == 2:
                mask_b = mask_b.unsqueeze(0)  # Add batch dimension if missing

            # Check if mask_b size matches image_b
            if mask_b.shape[1:3] != image_b.shape[1:3]:
                raise ValueError(f"Mask B dimensions ({mask_b.shape[1:3]}) do not match Image B dimensions ({image_b.shape[1:3]})")

        # Create a buffer for mask_a as well
        buffer_mask_a = torch.zeros((batch_a, height_b, width_b), device=mask_a.device)

        # Copy mask_a to buffer with same offsets as image_a
        if copy_height > 0 and copy_width > 0:
            buffer_mask_a[:,
                          dst_y_start:dst_y_start + copy_height,
                          dst_x_start:dst_x_start + copy_width] = mask_a[:,
                                                                         src_y_start:src_y_start + copy_height,
                                                                         src_x_start:src_x_start + copy_width]

        # Prepare variables for calculations
        a = buffer_mask_a.unsqueeze(-1)  # Add channel dimension for calculations
        b = mask_b.unsqueeze(-1)  # Add channel dimension for calculations
        A = buffer_a
        B = image_b

        # Если выбран premult, предварительно умножаем A на его маску
        if premult_a:
            A = A * a

        if premult_b:
            B = B * b

        # Apply selected compositing operation
        if operation == "atop":
            # Ab+B(1-a)
            result = A * a * mix + B * (1 - a * mix)
            mix_in_ops = True

        elif operation == "average":
            # (A+B)/2
            result = (A + B) / 2

        elif operation == "color":
            # Color blend mode keeps luminance of B but takes hue and saturation from A

            # Convert RGB to Lab используя наш метод
            A_lab = rgb2lab(A)
            B_lab = rgb2lab(B)

            # Create result Lab image taking L from B and a,b from A
            result_lab = torch.cat([
                B_lab[:,:,:,0:1],   # Lightness from B (сохраняем размерность)
                A_lab[:,:,:,1:2],   # a* from A
                A_lab[:,:,:,2:3]    # b* from A
            ], dim=3)

            # Convert back to RGB
            result = lab2rgb(result_lab)

            # Apply alpha blending based on mask
            if a is not None:
                result = result * a * mix + B * (1 - a * mix)
                mix_in_ops = True

        elif operation == "color-burn":
            # darken B towards A
            # When A is 0, result is 0. When A is 1, result is B
            # Formula: 1 - (1 - B) / A
            epsilon = 1e-7  # To avoid division by zero
            result = 1 - (1 - B) / (A + epsilon)
            result = torch.clamp(result, 0, 1)

        elif operation == "color-dodge":
            # brighten B towards A
            # Formula: B / (1 - A)
            epsilon = 1e-7  # To avoid division by zero
            result = B / (1 - A + epsilon)
            result = torch.clamp(result, 0, 1)

        elif operation == "conjoint-over":
            # A+B(1-a)/b, A if a>b
            epsilon = 1e-7  # To avoid division by zero
            result = torch.where(
                a * mix > b * mix,
                A,
                A + B * (1 - a * mix) / (b * mix + epsilon)
            )
            mix_in_ops = True

        elif operation == "copy":
            # A
            result = A

        elif operation == "difference":
            # abs(A-B)
            result = torch.abs(A - B)

        elif operation == "disjoint-over":
            # A+B(1-a)/b, A+B if a+b<1
            epsilon = 1e-7  # To avoid division by zero
            result = torch.where(
                (a * mix + b * mix) < 1,
                A + B,
                A + B * (1 - a * mix) / (b * mix + epsilon)
            )
            mix_in_ops = True

        elif operation == "divide":
            # A/B, 0 if A<0 and B<0
            epsilon = 1e-7  # To avoid division by zero
            result = torch.where(
                (A < 0) & (B < 0),
                torch.zeros_like(A),
                A / (B + epsilon)
            )
            result = torch.clamp(result, 0, 1)

        elif operation == "exclusion":
            # A+B-2AB
            result = A + B - 2 * A * B

        elif operation == "from":
            # B-A
            result = B - A
            result = torch.clamp(result, 0, 1)

        elif operation == "geometric":
            # 2AB/(A+B)
            epsilon = 1e-7  # To avoid division by zero
            result = 2 * A * B / (A + B + epsilon)

        elif operation == "hard-light":
            # multiply if A<.5, screen if A>.5
            result = torch.where(
                A < 0.5,
                2 * A * B,
                1 - 2 * (1 - A) * (1 - B)
            )

        elif operation == "hypot":
            # diagonal: sqrt(A^2 + B^2)
            result = torch.sqrt(A**2 + B**2)
            result = torch.clamp(result, 0, 1)

        elif operation == "in":
            # Ab
            result = A * b * mix
            mix_in_ops = True

        elif operation == "mask":
            # Ba
            result = B * a * mix
            mix_in_ops = True

        elif operation == "matte":
            # Aa+B(1-a) (unpremultiplied over)
            result = A * a * mix + B * (1 - a * mix)
            mix_in_ops = True

        elif operation == "max":
            # max(A,B)
            result = torch.maximum(A, B)

        elif operation == "min":
            # min(A,B)
            result = torch.minimum(A, B)

        elif operation == "minus":
            # A-B
            result = A - B
            result = torch.clamp(result, 0, 1)

        elif operation == "multiply":
            # AB, A if A<0 and B<0
            result = torch.where(
                (A < 0) & (B < 0),
                A,
                A * B
            )

        elif operation == "out":
            # A(1-b)
            result = A * (1 - b)

        elif operation == "over":
            # A+B(1-a)
            result = A + B * (1 - a)

        elif operation == "overlay":
            # multiply if B<.5, screen if B>.5
            result = torch.where(
                B < 0.5,
                2 * A * B,
                1 - 2 * (1 - A) * (1 - B)
            )

        elif operation == "plus":
            # A+B
            result = A + B
            result = torch.clamp(result, 0, 1)

        elif operation == "screen":
            # A+B-AB if A and B between 0-1, else A if A>B else B
            condition = (A >= 0) & (A <= 1) & (B >= 0) & (B <= 1)
            result = torch.where(
                condition,
                A + B - A * B,
                torch.where(A > B, A, B)
            )

        elif operation == "soft-light":
            # B(2A+(B(1-AB))) if AB<1, 2AB otherwise
            condition = A * B < 1
            result = torch.where(
                condition,
                B * (2 * A + (B * (1 - A * B))),
                2 * A * B
            )
            result = torch.clamp(result, 0, 1)

        elif operation == "spectral-multiply":
            # Lab-space multiplication with cross-luminance effect

            # Convert RGB to Lab используя наш метод
            A_lab = rgb2lab(A)
            B_lab = rgb2lab(B)

            # Get L,a,b channels
            L_a = A_lab[:,:,:,0]
            a_a = A_lab[:,:,:,1]
            b_a = A_lab[:,:,:,2]

            L_b = B_lab[:,:,:,0]
            a_b = B_lab[:,:,:,1]
            b_b = B_lab[:,:,:,2]

            # Spectral multiplication
            Res_L = L_a * L_b
            Res_a = a_a * L_b + a_b * L_a
            Res_b = b_a * L_b + b_b * L_a

            # Combine channels back to Lab
            result_lab = torch.stack([Res_L, Res_a, Res_b], dim=-1)

            # Convert back to RGB
            result = lab2rgb(result_lab)

            # Apply mask blending
            if a is not None:
                result = result * a * mix + B * (1 - a * mix)
                mix_in_ops = True

        elif operation == "spectral-multiply(rec709)":
            # Lab-space multiplication with cross-luminance effect

            # Convert RGB to Lab используя наш метод
            A_lab = srgb_to_rec709(A)
            B_lab = srgb_to_rec709(B)

            # Get L,a,b channels
            L_a = A_lab[:,:,:,0]
            a_a = A_lab[:,:,:,1]
            b_a = A_lab[:,:,:,2]

            L_b = B_lab[:,:,:,0]
            a_b = B_lab[:,:,:,1]
            b_b = B_lab[:,:,:,2]

            # Spectral multiplication
            Res_L = L_a * L_b
            Res_a = a_a * L_b + a_b * L_a
            Res_b = b_a * L_b + b_b * L_a

            # Combine channels back to Lab
            result_lab = torch.stack([Res_L, Res_a, Res_b], dim=-1)

            # Convert back to RGB
            result = rec709_to_srgb(result_lab)

            # Apply mask blending
            if a is not None:
                result = result * a * mix + B * (1 - a * mix)
                mix_in_ops = True

        elif operation == "stencil":
            # B(1-a)
            result = B * (1 - a * mix)
            mix_in_ops = True

        elif operation == "under":
            # A(1-b)+B
            result = A * (1 - b * mix) + B
            mix_in_ops = True

        elif operation == "xor":
            # A(1-b)+B(1-a)
            result = A * (1 - b * mix) + B * (1 - a * mix)
            mix_in_ops = True

        else:
            raise ValueError(f"Unknown operation: {operation}")

        result_mask = a + b * (1.0 - a)

        # Убираем размерность каналов для масок
        result_mask = result_mask.squeeze(-1)
        a_mask = a.squeeze(-1)
        b_mask = b.squeeze(-1)

        if not mix_in_ops:
            result = torch.lerp(B, result, mix)

        return (result, a_mask, b_mask, result_mask,)

IMAGECOMPOSE_CLASS_MAPPINGS = {
    "ImageCompositionX": ImageCompositionX,
}

IMAGECOMPOSE_NAME_MAPPINGS = {
    "ImageCompositionX": "Image Composition X",
}
