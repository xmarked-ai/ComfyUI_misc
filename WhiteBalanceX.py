from nodes import MAX_RESOLUTION
import torch
import math
import numpy as np

def cct_to_rgb(temperature):
    """Convert color temperature in Kelvin to RGB values"""
    # Algorithm based on Neil Bartlett's implementation
    temp = temperature / 100.0

    # Calculate red
    if temp <= 66:
        red = 255
    else:
        red = temp - 60
        red = 329.698727446 * (red ** -0.1332047592)
        red = max(0, min(255, red))

    # Calculate green
    if temp <= 66:
        green = temp
        green = 99.4708025861 * math.log(green) - 161.1195681661
    else:
        green = temp - 60
        green = 288.1221695283 * (green ** -0.0755148492)
    green = max(0, min(255, green))

    # Calculate blue
    if temp >= 66:
        blue = 255
    elif temp <= 19:
        blue = 0
    else:
        blue = temp - 10
        blue = 138.5177312231 * math.log(blue) - 305.0447927307
        blue = max(0, min(255, blue))

    return red/255, green/255, blue/255

def bradford_cat(image):
    """
    Implements Bradford chromatic adaptation transform.
    Automatically detects the source white point from the image
    and adapts to D65 standard illuminant.
    """
    # Bradford transformation matrix
    bradford_matrix = torch.tensor([
        [ 0.8951,  0.2664, -0.1614],
        [-0.7502,  1.7135,  0.0367],
        [ 0.0389, -0.0685,  1.0296]
    ], dtype=torch.float32)

    # Inverse Bradford transformation matrix
    bradford_matrix_inv = torch.tensor([
        [ 0.9869929, -0.1470543,  0.1599627],
        [ 0.4323053,  0.5183603,  0.0492912],
        [-0.0085287,  0.0400428,  0.9684867]
    ], dtype=torch.float32)

    # D65 white point (standard daylight)
    d65_xyz = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32)

    # Estimate source white point using top percentile pixels
    # This is more robust than using max values
    r_vals = image[0, :, :, 0].flatten()
    g_vals = image[0, :, :, 1].flatten()
    b_vals = image[0, :, :, 2].flatten()

    # Get top 5% brightest pixels
    r_thresh = torch.quantile(r_vals, 0.95)
    g_thresh = torch.quantile(g_vals, 0.95)
    b_thresh = torch.quantile(b_vals, 0.95)

    # Calculate source white point as average of bright pixels
    r_bright = r_vals[r_vals >= r_thresh].mean()
    g_bright = g_vals[g_vals >= g_thresh].mean()
    b_bright = b_vals[b_vals >= b_thresh].mean()

    # Convert source RGB to XYZ using standard sRGB matrix
    rgb_to_xyz = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=torch.float32)

    source_rgb = torch.tensor([r_bright, g_bright, b_bright], dtype=torch.float32)
    source_xyz = torch.matmul(rgb_to_xyz, source_rgb)

    # Normalize white point
    source_xyz = source_xyz / source_xyz[1]  # Normalize Y to 1.0

    # Convert source and destination white points to cone response domain
    source_lms = torch.matmul(bradford_matrix, source_xyz)
    d65_lms = torch.matmul(bradford_matrix, d65_xyz)

    # Compute the cone response transform ratios
    cat_ratio = d65_lms / source_lms

    # Calculate scaling factors for RGB channels
    # This is a simplification - we're approximating the Bradford transform
    # by directly scaling RGB instead of going through XYZ->LMS->scaled LMS->XYZ->RGB
    scale_r = cat_ratio[0]
    scale_g = cat_ratio[1]
    scale_b = cat_ratio[2]

    return scale_r, scale_g, scale_b

def retinex_wb(image):
    # Логарифмическое преобразование для разделения освещения
    log_image = torch.log(image[0] + 1e-6)  # Добавляем малую константу для избежания log(0)

    # Среднее по каналам как оценка освещения
    illumination_r = torch.mean(log_image[:, :, 0])
    illumination_g = torch.mean(log_image[:, :, 1])
    illumination_b = torch.mean(log_image[:, :, 2])

    # Корректируем, предполагая, что среднее освещение нейтрально
    scale_r = 1.0 / torch.exp(illumination_r) if illumination_r != 0 else 1.0
    scale_g = 1.0 / torch.exp(illumination_g) if illumination_g != 0 else 1.0
    scale_b = 1.0 / torch.exp(illumination_b) if illumination_b != 0 else 1.0

    return scale_r, scale_g, scale_b


from sklearn.cluster import KMeans

def dynamic_threshold_wb(image):
    # Преобразуем изображение в массив пикселей (N, 3)
    pixels = image[0].reshape(-1, 3).cpu().numpy()

    # Кластеризация с K-means (например, 5 кластеров)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(pixels)
    cluster_centers = kmeans.cluster_centers_

    # Находим кластер, где R, G, B наиболее близки (нейтральный)
    neutrality = np.abs(cluster_centers[:, 0] - cluster_centers[:, 1]) + \
                 np.abs(cluster_centers[:, 1] - cluster_centers[:, 2]) + \
                 np.abs(cluster_centers[:, 2] - cluster_centers[:, 0])
    neutral_cluster = cluster_centers[np.argmin(neutrality)]

    # Вычисляем масштабирующие коэффициенты
    scale_r = 1.0 / neutral_cluster[0] if neutral_cluster[0] > 0 else 1.0
    scale_g = 1.0 / neutral_cluster[1] if neutral_cluster[1] > 0 else 1.0
    scale_b = 1.0 / neutral_cluster[2] if neutral_cluster[2] > 0 else 1.0

    return torch.tensor([scale_r, scale_g, scale_b], dtype=torch.float32)


def perfect_reflector(image):
    # Find maximum values for each channel
    max_r = torch.max(image[0, :, :, 0])
    max_g = torch.max(image[0, :, :, 1])
    max_b = torch.max(image[0, :, :, 2])

    # Calculate scaling factors (assuming 1.0 is max)
    scale_r = 1.0 / max_r if max_r > 0 else 1.0
    scale_g = 1.0 / max_g if max_g > 0 else 1.0
    scale_b = 1.0 / max_b if max_b > 0 else 1.0

    return scale_r, scale_g, scale_b


def neutral_patches_wb(image, patch_size=16, neutral_quantile=0.1):
    # Разбиваем изображение на патчи
    h, w = image.shape[1], image.shape[2]
    if h < patch_size or w < patch_size:
        # Если изображение слишком маленькое, возвращаем нейтральные коэффициенты
        return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

    # Используем unfold для создания патчей
    patches = image[0].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    patches = patches.reshape(-1, 3, patch_size, patch_size)  # [N_patches, 3, patch_size, patch_size]

    # Вычисляем средний цвет каждого патча
    patch_means = patches.mean(dim=(2, 3))  # [N_patches, 3]

    # Оцениваем нейтральность: сумма абсолютных разниц между R, G, B
    neutrality = torch.abs(patch_means[:, 0] - patch_means[:, 1]) + \
                torch.abs(patch_means[:, 1] - patch_means[:, 2]) + \
                torch.abs(patch_means[:, 2] - patch_means[:, 0])

    # Выбираем топ neutral_quantile самых нейтральных патчей
    neutral_threshold = torch.quantile(neutrality, neutral_quantile)
    neutral_patches = patch_means[neutrality <= neutral_threshold]

    # Если нет нейтральных патчей, возвращаем нейтральные коэффициенты
    if len(neutral_patches) == 0:
        return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

    # Средний цвет нейтральных патчей как белая точка
    white_point = neutral_patches.mean(dim=0)

    # Масштабирующие коэффициенты
    scale_r = 1.0 / white_point[0] if white_point[0] > 0 else 1.0
    scale_g = 1.0 / white_point[1] if white_point[1] > 0 else 1.0
    scale_b = 1.0 / white_point[2] if white_point[2] > 0 else 1.0

    return torch.tensor([scale_r, scale_g, scale_b], dtype=torch.float32)


def histogram_neutral_wb(image, bins=128):
    # Создаём гистограмму для каждого канала
    r_vals = image[0, :, :, 0].flatten()
    g_vals = image[0, :, :, 1].flatten()
    b_vals = image[0, :, :, 2].flatten()

    # Вычисляем гистограммы
    hist_r, bins_r = torch.histogram(r_vals, bins=bins, range=(0.1, 0.9))
    hist_g, bins_g = torch.histogram(g_vals, bins=bins, range=(0.1, 0.9))
    hist_b, bins_b = torch.histogram(b_vals, bins=bins, range=(0.1, 0.9))

    # Находим области, где R, G, B близки (нейтральные)
    min_hist = torch.min(torch.stack([hist_r, hist_g, hist_b]), dim=0)[0]
    max_bin = torch.argmax(min_hist)  # Бин с максимальным пересечением
    bin_center = (bins_r[max_bin] + bins_r[max_bin + 1]) / 2  # Центр бина

    # Предполагаем, что центр этого бина — нейтральная точка
    white_point = torch.tensor([bin_center, bin_center, bin_center], dtype=torch.float32)

    # Масштабирующие коэффициенты
    scale_r = 1.0 / white_point[0] if white_point[0] > 0 else 1.0
    scale_g = 1.0 / white_point[1] if white_point[1] > 0 else 1.0
    scale_b = 1.0 / white_point[2] if white_point[2] > 0 else 1.0

    return scale_r, scale_g, scale_b


class WhiteBalanceX:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "use_CCT": ("BOOLEAN", {"default": False, "label_on": "Yes", "label_off": "No"}),
                "temperature": ("INT", {"default": 6500, "min": 1000, "max": 15000, "step": 50, "display": "slider"}),
                "color_red": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.00001}),
                "color_green": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.00001}),
                "color_blue": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.00001}),
                "color_space": ("BOOLEAN", {"default": False, "label_on": "RGB", "label_off": "XYZ", "forceInput": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("IMAGE_RGB", "IMAGE_XYZ")
    FUNCTION = "whitebalance"
    CATEGORY = "xmtools/nodes"

    def whitebalance(self, image, use_CCT, color_red, color_green, color_blue, temperature, color_space):
        # Get white balance values based on selected mode
        if use_CCT:
            color_red, color_green, color_blue = cct_to_rgb(temperature)

        # Остальная часть метода без изменений...

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

            image_RGB[0, :, :, 0] = 1 / color_red * image_RGB[0, :, :, 0]
            image_RGB[0, :, :, 1] = 1 / color_green * image_RGB[0, :, :, 1]
            image_RGB[0, :, :, 2] = 1 / color_blue * image_RGB[0, :, :, 2]

            Ln = image_RGB[0, :, :, 0] * 0.299 + image_RGB[0, :, :, 1] * 0.587 + image_RGB[0, :, :, 2] * 0.114

            balanced_image_RGB = image_RGB

            balanced_image_RGB[0, :, :, 0] = image_RGB[0, :, :, 0] / Ln * L
            balanced_image_RGB[0, :, :, 1] = image_RGB[0, :, :, 1] / Ln * L
            balanced_image_RGB[0, :, :, 2] = image_RGB[0, :, :, 2] / Ln * L

        else:
            # red_color, green_color, blue_color to XYZ as sample tensor
            rgb_values = torch.zeros(1, image_RGB.shape[1], image_RGB.shape[2], 3, dtype=torch.float32)
            rgb_values[0, :, :, 0] = color_red
            rgb_values[0, :, :, 1] = color_green
            rgb_values[0, :, :, 2] = color_blue
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

WHITEBALANCE_CLASS_MAPPINGS = {
    "WhiteBalanceX": WhiteBalanceX,
}

WHITEBALANCE_NAME_MAPPINGS = {
    "WhiteBalanceX": "White Balance X",
}
