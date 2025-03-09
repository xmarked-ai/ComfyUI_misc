import torch
import comfy.utils
import comfy.model_management
import comfy.sd
import comfy.lora
from comfy.model_patcher import ModelPatcher
import folder_paths
import re
import os
import copy
import logging

class SelectiveDepthLoraBlocksX:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
            "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),

            # Основные группы компонентов по функциональности
            "attn_blocks_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "Сила применения блоков внимания - влияют на пространственное восприятие"}),
            "modulation_blocks_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                               "tooltip": "Сила применения блоков модуляции - влияют на стилистику"}),
            "mlp_blocks_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "Сила применения MLP блоков - влияют на общее понимание содержания"}),

            # Разделение по типу данных
            "img_blocks_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Сила применения блоков для изображений"}),
            "txt_blocks_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Сила применения блоков для текста"}),

            # Уровни модели (разные глубины блоков имеют разное влияние)
            "early_blocks_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01,
                                           "tooltip": "Сила применения ранних блоков - влияют на низкоуровневые особенности"}),
            "middle_blocks_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01,
                                             "tooltip": "Сила применения средних блоков"}),
            "deep_blocks_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "Сила применения глубоких блоков - влияют на высокоуровневые особенности"}),

            # Особые компоненты
            "input_projection_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01,
                                               "tooltip": "Сила применения входных проекций"}),
            "output_projection_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01,
                                                "tooltip": "Сила применения выходных проекций"})
        }}

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply_selective_lora"
    CATEGORY = "xmtools/nodes"

    def _get_component_type(self, key):
        """Определяет тип компонента на основе его ключа"""
        if "attn" in key:
            return "attn"
        elif "mod" in key or "modulation" in key:
            return "modulation"
        elif "mlp" in key or "linear" in key:
            return "mlp"
        else:
            return "other"

    def _get_component_data_type(self, key):
        """Определяет тип данных (img/txt) компонента на основе его ключа"""
        if "img" in key:
            return "img"
        elif "txt" in key:
            return "txt"
        else:
            return "general"

    def _get_component_depth(self, key):
        """Определяет глубину компонента в модели"""
        # Извлекаем числовой индекс блока, если он есть
        block_num = -1
        match = re.search(r'blocks\.(\d+)', key)
        if not match:
            match = re.search(r'blocks\.(\d+)', key)
        if match:
            block_num = int(match.group(1))

        if block_num == -1:
            # Если нет номера блока, проверяем контекст
            if "in" in key or key.startswith("input") or "embedding" in key:
                return "early"
            elif "out" in key or key.startswith("output") or "final" in key:
                return "deep"
            else:
                return "middle"
        else:
            # Делим блоки на три группы по глубине
            total_blocks = 38  # Примерное общее количество блоков в FLUX
            if block_num < total_blocks // 3:
                return "early"
            elif block_num < 2 * (total_blocks // 3):
                return "middle"
            else:
                return "deep"

    def _is_input_projection(self, key):
        """Определяет, является ли компонент входной проекцией"""
        input_patterns = [r"txt_in", r"img_in", r"vector_in", r"time_in", r"guidance_in"]
        return any(re.search(pattern, key) for pattern in input_patterns)

    def _is_output_projection(self, key):
        """Определяет, является ли компонент выходной проекцией"""
        output_patterns = [r"final_layer", r"out\."]
        return any(re.search(pattern, key) for pattern in output_patterns)

    def _get_scale_for_key(self, key, config):
        """Вычисляет общий коэффициент масштабирования для компонента на основе всех параметров"""
        component_type = self._get_component_type(key)
        data_type = self._get_component_data_type(key)
        depth = self._get_component_depth(key)

        # Базовый коэффициент для всех компонентов
        scale = 1.0

        # Применяем масштабирование по типу компонента
        if component_type == "attn":
            scale *= config["attn_blocks_strength"]
        elif component_type == "modulation":
            scale *= config["modulation_blocks_strength"]
        elif component_type == "mlp":
            scale *= config["mlp_blocks_strength"]

        # Применяем масштабирование по типу данных
        if data_type == "img":
            scale *= config["img_blocks_strength"]
        elif data_type == "txt":
            scale *= config["txt_blocks_strength"]

        # Применяем масштабирование по глубине
        if depth == "early":
            scale *= config["early_blocks_strength"]
        elif depth == "middle":
            scale *= config["middle_blocks_strength"]
        elif depth == "deep":
            scale *= config["deep_blocks_strength"]

        # Применяем специальные коэффициенты для входных и выходных проекций
        if self._is_input_projection(key):
            scale *= config["input_projection_strength"]
        elif self._is_output_projection(key):
            scale *= config["output_projection_strength"]

        return scale

    def _custom_load_lora(self, lora_path, lora_sd, base_strength, config):
        """Создает модифицированную копию LoRA весов с примененными селективными коэффициентами"""
        # Создаем глубокую копию весов, чтобы не изменять оригинал
        modified_lora_sd = {}

        # Для отслеживания по типам
        applied_scales = {}

        # Модифицируем каждый тензор в соответствии с селективным масштабированием
        for key, value in lora_sd.items():
            # Пропускаем не-lora ключи
            if ".lora_" not in key:
                modified_lora_sd[key] = value.clone() if isinstance(value, torch.Tensor) else value
                continue

            # Получаем базовый ключ (без суффикса lora_A/lora_B)
            base_key = key.split(".lora_")[0]

            # Вычисляем селективный масштабный коэффициент
            selective_scale = self._get_scale_for_key(base_key, config)

            # Регистрируем примененный масштаб для логирования
            component_type = self._get_component_type(base_key)
            applied_scales[component_type] = applied_scales.get(component_type, []) + [selective_scale]

            # Применяем селективный масштаб
            if isinstance(value, torch.Tensor):
                # Умножаем на базовую силу и селективный масштаб
                modified_value = value.clone() * (selective_scale * base_strength)
                modified_lora_sd[key] = modified_value
            else:
                modified_lora_sd[key] = value

        # Логируем средние значения примененных масштабов по типам
        print("Applied selective scaling to LoRA weights:")
        for component_type, scales in applied_scales.items():
            avg_scale = sum(scales) / len(scales) if scales else 0
            print(f"  {component_type}: average scale = {avg_scale:.3f} (from {len(scales)} components)")

        return modified_lora_sd

    def apply_selective_lora(self, model, clip, lora_name, lora_strength,
                             attn_blocks_strength, modulation_blocks_strength, mlp_blocks_strength,
                             img_blocks_strength, txt_blocks_strength,
                             early_blocks_strength, middle_blocks_strength, deep_blocks_strength,
                             input_projection_strength, output_projection_strength):
        config = {
            "attn_blocks_strength": attn_blocks_strength,
            "modulation_blocks_strength": modulation_blocks_strength,
            "mlp_blocks_strength": mlp_blocks_strength,
            "img_blocks_strength": img_blocks_strength,
            "txt_blocks_strength": txt_blocks_strength,
            "early_blocks_strength": early_blocks_strength,
            "middle_blocks_strength": middle_blocks_strength,
            "deep_blocks_strength": deep_blocks_strength,
            "input_projection_strength": input_projection_strength,
            "output_projection_strength": output_projection_strength
        }

        try:
            # Загрузка исходных LoRA весов
            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)

            # Добавь конвертацию LoRA, как в стандартной функции
            lora_sd = comfy.lora_convert.convert_lora(lora_sd)

            # Создаем модифицированную версию LoRA с селективными масштабами
            modified_lora_sd = self._custom_load_lora(lora_path, lora_sd, lora_strength, config)

            # Применение модифицированных LoRA весов
            key_map = {}
            if model is not None:
                key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
            if clip is not None:
                key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

            # Выведи ключи для отладки
            # print("Keys in modified_lora_sd:", list(modified_lora_sd.keys()))
            # print("Keys in key_map:", list(key_map.keys()))

            # Загружаем LoRA с поддержкой ключевой карты
            loaded = comfy.lora.load_lora(modified_lora_sd, key_map)

            # Выведи ключи из loaded для отладки
            # print("Keys in loaded:", list(loaded.keys()))

            # Применяем патчи к модели и CLIP
            if model is not None:
                new_model = model.clone()
                model_keys = new_model.add_patches(loaded, 1.0)  # Уже умножили на lora_strength в _custom_load_lora
            else:
                new_model = None
                model_keys = set()

            if clip is not None:
                new_clip = clip.clone()
                clip_keys = new_clip.add_patches(loaded, 1.0)  # Уже умножили на lora_strength в _custom_load_lora
            else:
                new_clip = None
                clip_keys = set()

            # Проверка на неиспользованные ключи
            model_keys_set = set(model_keys)
            clip_keys_set = set(clip_keys)
            for key in loaded:
                if (key not in model_keys_set) and (key not in clip_keys_set):
                    print(f"WARNING: LoRA key not used: {key}")

            return (new_model, new_clip)

        except Exception as e:
            print(f"Error applying selective LoRA: {str(e)}")
            return (model, clip)

# Регистрация класса
DEPTHLORABLOCKSX_CLASS_MAPPINGS = {
    "SelectiveDepthLoraBlocksX": SelectiveDepthLoraBlocksX,
}

DEPTHLORABLOCKSX_NAME_MAPPINGS = {
    "SelectiveDepthLoraBlocksX": "Selective Depth Lora Blocks X",
}