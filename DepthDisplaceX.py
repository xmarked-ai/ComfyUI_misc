import torch
from torchvision import transforms
import torch.nn.functional as F
import PIL.Image
import numpy as np
import math

class DepthDisplaceX:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "strength": ("FLOAT", {"default": 20.0, "min": 0, "max": 100, "step": 0.1}),
                "blur_depth": ("FLOAT", {"default": 0, "min": 0, "max": 50, "step": 0.1}),
                "offset_x": ("INT", {"default": 0, "min": -2000, "max": 2000, "step": 1}),
                "offset_y": ("INT", {"default": 0, "min": -2000, "max": 2000, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
                "background": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "MASK", "composed_image", "offset_x", "offset_y",)
    FUNCTION = "depthdisplace"
    CATEGORY = "xmtools/nodes"

    def depthdisplace(self, image, depth_map, strength, blur_depth, offset_x=0, offset_y=0, mask=None, background=None):
        # Переставляем размерности image в [batch, channels, height, width]
        image = image.permute(0, 3, 1, 2)
        depth_map = depth_map.permute(0, 3, 1, 2)

        # Подготавливаем карту глубины
        # Берем первый канал и нормализуем
        depth = depth_map[:, 0:1, :, :]  # Берем только один канал

        # Если задано размытие, применяем его
        if blur_depth > 0:
            # Размер ядра должен быть нечетным
            kernel_size = int(blur_depth * 2) | 1  # Битовое ИЛИ с 1 делает число нечетным
            kernel_size = max(3, kernel_size)  # Минимальный размер ядра 3
            sigma = blur_depth * 0.5
            depth = transforms.GaussianBlur(kernel_size, sigma=sigma)(depth)

        # Объединяем image и mask если есть
        if mask is not None:
            mask = 1 - mask
            rgba = torch.cat([image, mask.unsqueeze(1)], dim=1)
        else:
            alpha = 1 - torch.ones((image.shape[0], 1, image.shape[2], image.shape[3]), device=image.device)
            rgba = torch.cat([image, alpha], dim=1)

        # Создаем тензор-контейнер размером с depth_map
        container = torch.zeros((rgba.shape[0], rgba.shape[1], depth_map.shape[2], depth_map.shape[3]), device=image.device)

        # Вычисляем начальные координаты для центрирования
        start_x = (depth_map.shape[3] - rgba.shape[3]) // 2
        start_y = (depth_map.shape[2] - rgba.shape[2]) // 2

        # Применяем смещения
        start_x += offset_x
        start_y += offset_y

        # Вычисляем валидные координаты для копирования
        valid_start_x = max(0, start_x)
        valid_start_y = max(0, start_y)
        valid_end_x = min(depth_map.shape[3], start_x + rgba.shape[3])
        valid_end_y = min(depth_map.shape[2], start_y + rgba.shape[2])

        # Вычисляем соответствующие координаты в исходном изображении
        src_start_x = max(0, -start_x)
        src_start_y = max(0, -start_y)
        src_end_x = src_start_x + (valid_end_x - valid_start_x)
        src_end_y = src_start_y + (valid_end_y - valid_start_y)

        # Копируем валидную часть изображения
        if (src_end_x > src_start_x) and (src_end_y > src_start_y):
            container[:, :, valid_start_y:valid_end_y, valid_start_x:valid_end_x] = \
                rgba[:, :, src_start_y:src_end_y, src_start_x:src_end_x]

        # Создаем сетку координат для displacement
        height, width = depth_map.shape[2:]
        grid_y, grid_x = torch.meshgrid(torch.arange(height, device=depth.device),
                                      torch.arange(width, device=depth.device),
                                      indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1).float().unsqueeze(0)

        # Применяем смещение на основе карты глубины
        displacement = depth * strength
        grid[:, :, :, 0] += displacement.squeeze(1)  # Смещение по X
        grid[:, :, :, 1] += displacement.squeeze(1)  # Смещение по Y

        # Нормализуем координаты сетки
        grid[:, :, :, 0] = (grid[:, :, :, 0] / (width - 1)) * 2 - 1
        grid[:, :, :, 1] = (grid[:, :, :, 1] / (height - 1)) * 2 - 1

        # Ограничиваем координаты
        grid = torch.clamp(grid, min=-1, max=1)

        # Применяем displacement к container используя grid_sample
        displaced = F.grid_sample(container, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        # Разделяем обратно на image и mask
        displaced_image = displaced[:, :3, :, :]
        displaced_mask = displaced[:, 3:, :, :]

        # Возвращаем размерности в исходный порядок
        displaced_image = displaced_image.permute(0, 2, 3, 1)

        # Подготавливаем composed_image
        if background is not None:
            # Переводим background в нужный формат если нужно
            if len(background.shape) == 3:
                background = background.unsqueeze(0)

            # Ресайзим displaced_image и маску под размер background
            displaced_image_resized = F.interpolate(
                displaced_image.permute(0, 3, 1, 2),  # [B, C, H, W]
                size=(background.shape[1], background.shape[2]),
                mode='bilinear',
                align_corners=True
            ).permute(0, 2, 3, 1)  # Возвращаем в [B, H, W, C]

            displaced_mask_resized = F.interpolate(
                displaced_mask,  # Уже в формате [B, 1, H, W]
                size=(background.shape[1], background.shape[2]),
                mode='bilinear',
                align_corners=True
            )

            # Используем маску для композитинга
            alpha = displaced_mask_resized.squeeze(1).unsqueeze(-1)  # [B, H, W, 1]
            composed_image = displaced_image_resized * alpha + background * (1 - alpha)
        else:
            composed_image = displaced_image.detach().clone()

        return (displaced_image, displaced_mask.squeeze(1), composed_image, offset_x, offset_y)


DEPTHDISPLACE_CLASS_MAPPINGS = {
    "DepthDisplaceX": DepthDisplaceX,
}

DEPTHDISPLACE_NAME_MAPPINGS = {
    "DepthDisplaceX": "Depth Displace X",
}
