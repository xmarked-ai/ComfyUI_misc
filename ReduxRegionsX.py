import torch
import comfy.ops
from comfy.ldm.flux.redux import ReduxImageEncoder
import math

ops = comfy.ops.manual_cast

class ColorTransferNodeX:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "style_model": ("STYLE_MODEL",),
            "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            "color_weight": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.01,
                "tooltip": "Сила переноса цвета. 0 = без переноса, 1 = нормальный перенос, >1 = усиленный перенос"
            }),
            "color_region_only": ("BOOLEAN", {
                "default": True,
                "tooltip": "Если включено, будут модифицированы только области эмбеддинга, отвечающие за цвет"
            }),
            "preserve_lighting": ("FLOAT", {
                "default": 0.3,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "Сохранение оригинального освещения при переносе цвета"
            }),
            "saturation_boost": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 2.0,
                "step": 0.01,
                "tooltip": "Усиление насыщенности цветов"
            })
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_color_transfer"
    CATEGORY = "xmtools/nodes"

    def identify_color_region(self, feature_tensor):
        """
        Определяет регион в эмбеддингах, наиболее ответственный за цветовую информацию
        """
        feature_size = feature_tensor.shape[-1]

        # Основной регион цвета (примерно 1/5 от полного тензора)
        # В стандартном случае это вторая часть тензора (примерно 819-1638 индексы)
        # start_idx = feature_size // 5
        # end_idx = start_idx * 2
        start_idx = (feature_size // 20) * 5
        end_idx = (feature_size // 20 ) * 6

        # Дополнительные области, содержащие цветовую информацию
        # (части стилистического региона и текстурного)
        style_color_region = slice(0, start_idx // 3)
        texture_color_region = slice(feature_size - start_idx // 4, feature_size)

        color_regions = {
            'primary': slice(start_idx, end_idx),
            'style_color': style_color_region,
            'texture_color': texture_color_region
        }

        return color_regions

    def apply_color_transfer(self, conditioning, style_model, clip_vision_output,
                            color_weight=1.0, color_region_only=True,
                            preserve_lighting=0.3, saturation_boost=1.0):

        # Получаем эмбеддинги изображения-источника стиля
        image_cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1)

        # Получаем текстовые эмбеддинги из conditioning
        text_features = conditioning[0][0]

        # Определяем области эмбеддингов, ответственные за цвет
        color_regions = self.identify_color_region(image_cond)

        # Если нужно перенести только цвет, работаем только с цветовыми областями
        if color_region_only:
            # Создаем копию оригинальных эмбеддингов
            result_cond = image_cond.clone()

            # Применяем вес цвета к основной цветовой области
            primary_region = color_regions['primary']
            # Применяем насыщенность к цветовой области
            result_cond[..., primary_region] = image_cond[..., primary_region] * color_weight * saturation_boost

            # Для областей стиля и текстуры, которые влияют на цвет
            style_region = color_regions['style_color']
            texture_region = color_regions['texture_color']

            # Применяем меньший вес к дополнительным цветовым областям
            color_style_factor = color_weight * 0.7 * saturation_boost
            result_cond[..., style_region] = image_cond[..., style_region] * color_style_factor

            color_texture_factor = color_weight * 0.5 * saturation_boost
            result_cond[..., texture_region] = image_cond[..., texture_region] * color_texture_factor

            # Если нужно сохранить освещение, смешиваем с оригинальными эмбеддингами
            # Освещение обычно содержится в высокочастотной части эмбеддингов цвета
            if preserve_lighting > 0:
                # Определяем области, отвечающие за освещение (примерно в середине цветового региона)
                lighting_idx_start = primary_region.start + (primary_region.stop - primary_region.start) // 3
                lighting_idx_end = primary_region.start + (primary_region.stop - primary_region.start) * 2 // 3
                lighting_region = slice(lighting_idx_start, lighting_idx_end)

                # Смешиваем с оригинальными эмбеддингами для сохранения освещения
                result_cond[..., lighting_region] = (
                    result_cond[..., lighting_region] * (1 - preserve_lighting) +
                    image_cond[..., lighting_region] * preserve_lighting
                )
        else:
            # Если нужно применить цвет ко всему изображению, просто умножаем на вес
            result_cond = image_cond * color_weight

        # Подготавливаем результат в формате conditioning
        result_cond = result_cond.unsqueeze(dim=0)  # Восстанавливаем размерность batch

        # Создаем новый conditioning с модифицированными эмбеддингами
        c = []
        for t in conditioning:
            n = [torch.cat((t[0], result_cond), dim=1), t[1].copy()]
            c.append(n)

        return (c,)

class RegionTesterNodeX:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "style_model": ("STYLE_MODEL",),
            "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            "regions_number": ("INT", {
                "default": 10,
                "min": 2,
                "max": 100,
                "step": 1,
                "tooltip": "На сколько равных регионов разделить эмбеддинг"
            }),
            "start_region": ("INT", {
                "default": 0,
                "min": 0,
                "max": 99,
                "step": 1,
                "tooltip": "Начальный регион (от 0)"
            }),
            "end_region": ("INT", {
                "default": 1,
                "min": 0,
                "max": 99,
                "step": 1,
                "tooltip": "Конечный регион (включительно)"
            }),
            "weight": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.01,
                "tooltip": "Вес для выбранного региона"
            }),
            "other_regions_weight": ("FLOAT", {
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "Вес для остальных регионов (0 = отключить остальные регионы)"
            }),
            "debug_mode": ("BOOLEAN", {
                "default": False,
                "tooltip": "Выводить отладочную информацию о регионах"
            })
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_region_test"
    CATEGORY = "xmtools/nodes"

    def apply_region_test(self, conditioning, style_model, clip_vision_output,
                        regions_number=10, start_region=0, end_region=1,
                        weight=1.0, other_regions_weight=0.0, debug_mode=False):

        # Получаем эмбеддинги изображения-источника стиля
        image_cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1)

        # Получаем размер фичей
        feature_size = image_cond.shape[-1]

        # Создаем копию оригинальных эмбеддингов
        result_cond = image_cond.clone()

        # Вычисляем размер одного региона
        region_size = feature_size // regions_number

        # Вычисляем границы выбранного региона
        start_idx = region_size * start_region
        end_idx = region_size * (end_region + 1)

        # Убеждаемся, что индексы в допустимых пределах
        start_idx = max(0, min(start_idx, feature_size - 1))
        end_idx = max(0, min(end_idx, feature_size))

        if debug_mode:
            print(f"Размер эмбеддинга: {feature_size}")
            print(f"Количество регионов: {regions_number}")
            print(f"Размер одного региона: {region_size}")
            print(f"Выбранный диапазон регионов: {start_region}-{end_region}")
            print(f"Индексы региона: {start_idx}-{end_idx}")

        # Создаем маску для всего эмбеддинга
        if other_regions_weight > 0:
            # Применяем вес ко всему эмбеддингу
            result_cond = image_cond * other_regions_weight
        else:
            # Обнуляем весь эмбеддинг
            result_cond = torch.zeros_like(image_cond)

        # Применяем вес к выбранному региону
        result_cond[..., start_idx:end_idx] = image_cond[..., start_idx:end_idx] * weight

        # Подготавливаем результат в формате conditioning
        result_cond = result_cond.unsqueeze(dim=0)  # Восстанавливаем размерность batch

        # Создаем новый conditioning с модифицированными эмбеддингами
        c = []
        for t in conditioning:
            n = [torch.cat((t[0], result_cond), dim=1), t[1].copy()]
            c.append(n)

        return (c,)

REDUXREGIONSX_CLASS_MAPPINGS = {
    "ColorTransferNodeX": ColorTransferNodeX,
    "RegionTesterNodeX": RegionTesterNodeX,
}

REDUXREGIONSX_NAME_MAPPINGS = {
    "ColorTransferNodeX": "Color Transfer Node X",
    "RegionTesterNodeX": "RegionTesterNodeX",
}
