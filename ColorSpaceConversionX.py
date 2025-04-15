import torch
import torch.nn.functional as F
import numpy as np

def print_rgb_stats(tensor):
    # Для тензора формата (1, Y, X, 3)
    min_r = tensor[..., 0].min().item()
    max_r = tensor[..., 0].max().item()

    min_g = tensor[..., 1].min().item()
    max_g = tensor[..., 1].max().item()

    min_b = tensor[..., 2].min().item()
    max_b = tensor[..., 2].max().item()

    print(f"R канал: мин = {min_r:.6f}, макс = {max_r:.6f}")
    print(f"G канал: мин = {min_g:.6f}, макс = {max_g:.6f}")
    print(f"B канал: мин = {min_b:.6f}, макс = {max_b:.6f}")

def rgb_to_lab(rgb):
    # sRGB -> линейный RGB (гамма-коррекция)
    rgb_lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))

    # RGB -> XYZ (матрица из источника Линдблума)
    r, g, b = rgb_lin[..., 0], rgb_lin[..., 1], rgb_lin[..., 2]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # Опорные значения для D65
    x_n, y_n, z_n = 0.95047, 1.0, 1.08883

    # Нормализация XYZ
    x = x / x_n
    y = y / y_n
    z = z / z_n

    # Функция f для преобразования XYZ -> Lab
    epsilon = 0.008856
    kappa = 903.3

    fx = torch.where(x > epsilon, x.pow(1.0/3.0), (kappa * x + 16.0) / 116.0)
    fy = torch.where(y > epsilon, y.pow(1.0/3.0), (kappa * y + 16.0) / 116.0)
    fz = torch.where(z > epsilon, z.pow(1.0/3.0), (kappa * z + 16.0) / 116.0)

    # Вычисление Lab
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    # Нормализация: L к [0,1] и a,b к [-1,1]
    L = L / 100.0  # L теперь в [0,1]
    a = a / 127.0  # a теперь примерно в [-1,1]
    b = b / 127.0  # b теперь примерно в [-1,1]

    # Формирование тензора
    lab = torch.stack([L, a, b], dim=-1)
    # print_rgb_stats(lab)
    return lab

def lab_to_rgb(lab):
    # Денормализация
    L = lab[..., 0] * 100.0  # L обратно в [0,100]
    a = lab[..., 1] * 127.0  # a обратно в [-127,127]
    b = lab[..., 2] * 127.0  # b обратно в [-127,127]

    # Опорные значения для D65
    x_n, y_n, z_n = 0.95047, 1.0, 1.08883

    # Вычисление промежуточных значений
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    # Возведение в куб или линейная интерполяция
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

    # XYZ to RGB (обратная матрица)
    r = x *  3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y *  1.8760108 + z *  0.0415560
    b = x *  0.0556434 + y * -0.2040259 + z *  1.0572252

    # Линейный RGB -> sRGB
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

    # Формирование тензора
    rgb = torch.stack([r, g, b], dim=-1)
    return rgb

def srgb_to_rec709(image):
    threshold = 0.04045
    linear = torch.where(image <= threshold, image / 12.92, ((image + 0.055) / 1.055).pow(2.4))

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
    srgb = torch.where(linear <= threshold, linear * 12.92, 1.055 * linear.pow(1/2.4) - 0.055)

    return srgb

def rgb_to_cmyk(rgb, kk=0.15, ss=5.0, srgb=False):
    r = rgb[..., 0].clone()
    g = rgb[..., 1].clone()
    b = rgb[..., 2].clone()

    if srgb:
        r = lin2srgb(r)
        g = lin2srgb(g)
        b = lin2srgb(b)

    k = srgb_to_rec709(rgb)[..., 0].clone()
    k = (1 + kk) * torch.pow(1 - k, ss) - kk
    k = torch.clamp(k, 0.0, 1.0)

    c = 1 - r
    m = 1 - g
    y = 1 - b

    cmyk = torch.zeros((*rgb.shape[:-1], 4), device=rgb.device, dtype=rgb.dtype)
    k_mask = k < 1.0
    denominator = torch.where(k_mask, 1.0 - k, torch.ones_like(k))
    cmyk[..., 0] = torch.where(k_mask, (c - k) / denominator, torch.zeros_like(c))
    cmyk[..., 1] = torch.where(k_mask, (m - k) / denominator, torch.zeros_like(m))
    cmyk[..., 2] = torch.where(k_mask, (y - k) / denominator, torch.zeros_like(y))
    cmyk[..., 3] = k

    return cmyk

def cmyk_to_rgb(cmyk, srgb=False):
    c = cmyk[..., 0]
    m = cmyk[..., 1]
    y = cmyk[..., 2]
    k = cmyk[..., 3]

    r = (1 - k) * (1 - c)
    g = (1 - k) * (1 - m)
    b = (1 - k) * (1 - y)

    rgb = torch.zeros((*cmyk.shape[:-1], 3), device=cmyk.device, dtype=cmyk.dtype)
    rgb[..., 0] = r
    rgb[..., 1] = g
    rgb[..., 2] = b

    if srgb:
        rgb[..., 0] = srgb2lin(rgb[..., 0])
        rgb[..., 1] = srgb2lin(rgb[..., 1])
        rgb[..., 2] = srgb2lin(rgb[..., 2])

    return rgb

# Вспомогательные функции для конвертации lin <-> sRGB
def lin2srgb(x):
    """Convert linear RGB to sRGB"""
    return torch.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * torch.pow(x, 1.0/2.4) - 0.055
    )

def srgb2lin(x):
    """Convert sRGB to linear RGB"""
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        torch.pow((x + 0.055) / 1.055, 2.4)
    )

def rgb_to_hsv(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    max_rgb, argmax_rgb = torch.max(rgb, dim=-1)
    min_rgb, _ = torch.min(rgb, dim=-1)
    delta = max_rgb - min_rgb

    # Инициализируем HSV тензор
    hsv = torch.zeros_like(rgb)

    # Вычисляем H (hue)
    # Создаем маску для ненулевых delta
    delta_mask = delta > 0

    # Рассчитываем h для каждого случая в зависимости от того, какой канал максимальный
    h = torch.zeros_like(max_rgb)

    # Кейс: max_rgb = r
    r_mask = (argmax_rgb == 0) & delta_mask
    if torch.any(r_mask):
        h[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6

    # Кейс: max_rgb = g
    g_mask = (argmax_rgb == 1) & delta_mask
    if torch.any(g_mask):
        h[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2

    # Кейс: max_rgb = b
    b_mask = (argmax_rgb == 2) & delta_mask
    if torch.any(b_mask):
        h[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4

    h = h / 6.0  # Нормализуем в диапазон [0, 1]

    # Вычисляем S (saturation)
    s = torch.zeros_like(max_rgb)
    s[max_rgb > 0] = delta[max_rgb > 0] / max_rgb[max_rgb > 0]

    # Вычисляем V (value)
    v = max_rgb

    # Заполняем выходной HSV тензор
    hsv[..., 0] = h
    hsv[..., 1] = s
    hsv[..., 2] = v

    return hsv

def hsv_to_rgb(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Масштабируем h в диапазон [0, 6)
    h_scaled = h * 6.0

    # Вычисляем целую и дробную части
    i = torch.floor(h_scaled)
    f = h_scaled - i

    # Вычисляем вспомогательные значения
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    # Инициализируем RGB тензор
    rgb = torch.zeros_like(hsv)

    # Вычисляем RGB на основе сектора Hue
    h_mod = i.long() % 6

    # Создаем маски для каждого сектора
    mask_0 = (h_mod == 0)
    mask_1 = (h_mod == 1)
    mask_2 = (h_mod == 2)
    mask_3 = (h_mod == 3)
    mask_4 = (h_mod == 4)
    mask_5 = (h_mod == 5)

    # Заполняем RGB для каждого сектора
    rgb[..., 0] = torch.where(mask_0, v, torch.where(mask_1, q, torch.where(mask_2, p, torch.where(mask_3, p, torch.where(mask_4, t, v)))))
    rgb[..., 1] = torch.where(mask_0, t, torch.where(mask_1, v, torch.where(mask_2, v, torch.where(mask_3, q, torch.where(mask_4, p, p)))))
    rgb[..., 2] = torch.where(mask_0, p, torch.where(mask_1, p, torch.where(mask_2, t, torch.where(mask_3, v, torch.where(mask_4, v, q)))))

    return rgb



class ColorSpaceConversionX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "channels": ("IMAGE",),
                "conversion_type": (["RGB_to_HSV", "HSV_to_RGB",
                                    "RGB_to_LAB", "LAB_to_RGB",
                                    "RGB_to_YCbCr", "YCbCr_to_RGB",
                                    "RGB_to_CMYK", "CMYK_to_RGB"],),
            },
            "optional": {
                "k": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("channels", "k",)
    FUNCTION = "convert"
    CATEGORY = "xmtools/nodes"

    def convert(self, channels, conversion_type, k=None):
        # Убедимся, что формат входного тензора правильный (1, Y, X, 3)
        if len(channels.shape) != 4 or channels.shape[3] != 3:
            raise ValueError(f"Ожидается тензор формата (1, Y, X, 3), получен {channels.shape}")

        # Выбор функции конверсии на основе параметра conversion_type
        if conversion_type == "RGB_to_HSV":
            result = rgb_to_hsv(channels)
        elif conversion_type == "HSV_to_RGB":
            result = hsv_to_rgb(channels)
        elif conversion_type == "RGB_to_LAB":
            result = rgb_to_lab(channels)
        elif conversion_type == "LAB_to_RGB":
            result = lab_to_rgb(channels)
        elif conversion_type == "RGB_to_YCbCr":
            result = srgb_to_rec709(channels)
        elif conversion_type == "YCbCr_to_RGB":
            result = rec709_to_srgb(channels)
        elif conversion_type == "RGB_to_CMYK":
            cmyk = rgb_to_cmyk(channels)
            cmy = cmyk[..., :3]  # (1, Y, X, 3)
            k = cmyk[..., 3]     # (1, Y, X)
            return (cmy, k,)
        elif conversion_type == "CMYK_to_RGB":
            if k is None:
                    raise ValueError("Для CMYK_to_RGB требуется тензор k")
            cmyk = torch.cat((channels, k[..., None]), dim=-1)  # (1, Y, X, 4)
            result = cmyk_to_rgb(cmyk)
        else:
            raise ValueError(f"Неизвестный тип конверсии: {conversion_type}")

        return (result,)

# Регистрация ноды в ComfyUI
COLORSPACECONV_CLASS_MAPPINGS = {
    "ColorSpaceConversionX": ColorSpaceConversionX
}

COLORSPACECONV_DISPLAY_NAME_MAPPINGS = {
    "ColorSpaceConversionX": "Color Space Conversion X"
}