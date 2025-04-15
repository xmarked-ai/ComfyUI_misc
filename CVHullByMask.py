import numpy as np
from scipy.spatial import ConvexHull
import torch

class ConvexHullByMaskX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),  # Входная маска в формате тензора (1, Y, X)
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "process_mask"
    CATEGORY = "xmtools/nodes"

    def process_mask(self, mask):
        # Преобразуем входной тензор в numpy массив
        mask_np = mask.squeeze(0).cpu().numpy()  # Убираем размерность 1, получаем (Y, X)

        # Находим координаты всех белых точек (значение > 0)
        points = np.array(np.where(mask_np > 0)).T  # Получаем массив координат [(y1, x1), (y2, x2), ...]

        if len(points) < 3:
            # Если точек меньше 3, выпуклая оболочка не имеет смысла, возвращаем исходную маску
            return (mask,)

        # Строим выпуклую оболочку
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]  # Координаты вершин оболочки

        # Создаём пустую маску того же размера
        output_mask = np.zeros_like(mask_np)

        # Заполняем многоугольник
        # Для этого используем алгоритм заливки (например, через растеризацию)
        from skimage.draw import polygon
        rr, cc = polygon(hull_points[:, 0], hull_points[:, 1], shape=mask_np.shape)
        output_mask[rr, cc] = 1

        # Преобразуем обратно в тензор с нужной размерностью (1, Y, X)
        output_tensor = torch.from_numpy(output_mask).unsqueeze(0).float()

        return (output_tensor,)

# Регистрация ноды в ComfyUI
CVHULLMASKX_CLASS_MAPPINGS = {
    "ConvexHullByMaskX": ConvexHullByMaskX
}

CVHULLMASK_NAME_MAPPINGS = {
    "ConvexHullByMaskX": "Convex Hull by Mask X"
}