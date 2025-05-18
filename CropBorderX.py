import torch

class CropBorderX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Входной тензор изображения [1, Y, X, 3]
                "tolerance": ("FLOAT", {"default": 0.09, "min": 0.0, "max": 0.5, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_border"
    CATEGORY = "xmtools/nodes"

    def crop_border(self, image, tolerance):
        # Проверяем форму входного тензора
        if image.shape[0] != 1 or image.shape[3] != 3:
            raise ValueError("Expected image tensor of shape [1, Y, X, 3]")

        # Работаем с тензором [Y, X, 3]
        img_tensor = image[0]  # Убираем первую размерность: [Y, X, 3]

        # Берем цвет верхнего левого пикселя как цвет рамки
        border_color = img_tensor[0, 0]  # [3]

        # Получаем размеры изображения
        height, width = img_tensor.shape[:2]

        # Инициализируем границы
        top, bottom, left, right = 0, height - 1, 0, width - 1

        # Сканируем сверху вниз
        for i in range(height):
            row = img_tensor[i, :]  # [X, 3]
            # Проверяем, есть ли в строке пиксель, отличающийся от border_color
            diff = torch.abs(row - border_color) > tolerance  # [X, 3]
            if diff.any():  # Если есть хотя бы один отличающийся пиксель
                top = i
                break

        # Сканируем снизу вверх
        for i in range(height - 1, -1, -1):
            row = img_tensor[i, :]  # [X, 3]
            diff = torch.abs(row - border_color) > tolerance  # [X, 3]
            if diff.any():
                bottom = i
                break

        # Сканируем слева направо
        for j in range(width):
            col = img_tensor[:, j]  # [Y, 3]
            diff = torch.abs(col - border_color) > tolerance  # [Y, 3]
            if diff.any():
                left = j
                break

        # Сканируем справа налево
        for j in range(width - 1, -1, -1):
            col = img_tensor[:, j]  # [Y, 3]
            diff = torch.abs(col - border_color) > tolerance  # [Y, 3]
            if diff.any():
                right = j
                break

        # Проверяем, есть ли что обрезать
        if top >= bottom or left >= right:
            # Если обрезать не удалось, возвращаем исходное изображение
            return (image,)

        # Обрезаем изображение
        cropped_tensor = img_tensor[top:bottom + 1, left:right + 1, :]  # [Y', X', 3]
        cropped_tensor = cropped_tensor.unsqueeze(0)  # [1, Y', X', 3]

        return (cropped_tensor,)

NODE_CLASS_MAPPINGS = {
    "CropBorderX": CropBorderX
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CropBorderX": "Crop Border X"
}