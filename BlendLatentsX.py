import torch
import torch.nn.functional as F

class BlendLatentsX:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent1": ("LATENT", {"tooltip": "First latent to blend (e.g., from KSampler with first prompt)."}),
                "latent2": ("LATENT", {"tooltip": "Second latent to blend (e.g., from KSampler with second prompt)."}),
                "mask": ("MASK", {"tooltip": "Mask defining where latent1 dominates (white) and latent2 dominates (black)."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "blend"
    CATEGORY = "xmtools/nodes"
    DESCRIPTION = "Blends two LATENT tensors using a mask, where white areas favor latent1 and black areas favor latent2."

    def blend(self, latent1: dict, latent2: dict, mask: torch.Tensor) -> tuple[dict]:
        # Извлекаем тензоры из словарей LATENT
        samples1 = latent1["samples"]  # [batch, channels, latent_h, latent_w]
        samples2 = latent2["samples"]  # То же самое

        # Проверяем совместимость размеров
        assert samples1.shape == samples2.shape, f"Latent sizes must match, got {samples1.shape} and {samples2.shape}"

        # Приводим маску к размеру латентов
        latent_h, latent_w = samples1.shape[2], samples1.shape[3]
        mask = mask.squeeze()  # Убираем лишние размерности
        if len(mask.shape) == 3:
            mask = mask[0]  # Если [1, H, W]
        elif len(mask.shape) != 2:
            raise ValueError(f"Mask must be 2D or 3D with batch=1, got shape {mask.shape}")

        # Интерполируем маску до размера латентов
        mask_resized = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
            size=(latent_h, latent_w),
            mode="nearest"  # Или "bilinear" для плавных переходов
        ).squeeze(0)  # [1, latent_h, latent_w]

        # Убеждаемся, что маска в диапазоне [0, 1]
        mask_resized = torch.clamp(mask_resized, 0, 1)

        # Смешиваем латенты
        blended_samples = samples1 * mask_resized + samples2 * (1 - mask_resized)

        # Копируем словарь latent1 и заменяем samples
        result = latent1.copy()
        result["samples"] = blended_samples

        return (result,)

NODE_CLASS_MAPPINGS = {
    "BlendLatentsX": BlendLatentsX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlendLatentsX": "Blend LatentsX",
}
