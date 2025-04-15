import torch
import comfy.samplers
import comfy.model_management
import comfy.sample
import comfy.utils
import numpy as np
import math
from comfy.samplers import sampler_object
import latent_preview
import comfy.model_sampling
import comfy_extras.nodes_custom_sampler
import node_helpers

class RegionalPromptSamplerX:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "base_positive": ("CONDITIONING",),
                "base_negative": ("CONDITIONING",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler_ancestral"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "flux": ("BOOLEAN", {"default": False, "label_on": "Enable", "label_off": "Disable"}),
                "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 100.0, "step": 0.01}),
                "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.01}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "mask1": ("MASK",), "positive1": ("CONDITIONING",), "weight1": ("FLOAT", {"default": 1.0}),
                "regional_cfg1": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "mask_threshold1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask2": ("MASK",), "positive2": ("CONDITIONING",), "weight2": ("FLOAT", {"default": 1.0}),
                "regional_cfg2": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "mask_threshold2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask3": ("MASK",), "positive3": ("CONDITIONING",), "weight3": ("FLOAT", {"default": 1.0}),
                "regional_cfg3": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "mask_threshold3": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "xmtools/nodes"

    def sample(self, model, latent, base_positive, base_negative, steps, cfg, sampler_name, scheduler, seed, denoise, flux=False,
               max_shift=1.15, base_shift=0.5, guidance=3.5,
               mask1=None, positive1=None, weight1=1.0, regional_cfg1=None, mask_threshold1=0.5, denoise1=1.0,
               mask2=None, positive2=None, weight2=1.0, regional_cfg2=None, mask_threshold2=0.5, denoise2=1.0,
               mask3=None, positive3=None, weight3=1.0, regional_cfg3=None, mask_threshold3=0.5, denoise3=1.0):

        # Validate input latent
        if "samples" not in latent:
            raise ValueError("Invalid latent input: missing 'samples' key")

        # Collect regional prompts
        regional_prompts = []
        for mask, positive, weight, regional_cfg, mask_threshold, region_denoise in [
            (mask1, positive1, weight1, regional_cfg1, mask_threshold1, denoise1),
            (mask2, positive2, weight2, regional_cfg2, mask_threshold2, denoise2),
            (mask3, positive3, weight3, regional_cfg3, mask_threshold3, denoise3)
        ]:
            if mask is not None and positive is not None:
                regional_prompts.append((mask, positive, weight, regional_cfg, mask_threshold, region_denoise))

        try:
            # Run sampling
            result = self.regional_sample(
                model=model,
                latent=latent,
                base_positive=base_positive,
                base_negative=base_negative,
                regional_prompts=regional_prompts,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                seed=seed,
                denoise=denoise,
                flux=flux,
                max_shift=max_shift,
                base_shift=base_shift,
                guidance=guidance
            )
        except Exception as e:
            print(f"Regional sampling failed: {str(e)}")
            # Return original latent as fallback
            result = latent.copy()

        # Ensure valid output format
        if "samples" not in result:
            result = {"samples": latent["samples"]}

        return (result,)

    def regional_sample(self, model, latent, base_positive, base_negative, regional_prompts, steps, cfg, sampler_name, scheduler, seed, denoise, flux=False, max_shift=1.15, base_shift=0.5, guidance=3.5):
        try:
            # Оригинальная модель для ссылки
            original_model = model

            # Если Flux включен, клонируем модель и настраиваем параметры Flux
            if flux:
                # Клонируем модель для применения Flux
                m = model.clone()

                # Расчет параметра сдвига на основе размера изображения
                latent_h, latent_w = latent["samples"].shape[2:]
                x1 = 256
                x2 = 4096
                mm = (max_shift - base_shift) / (x2 - x1)
                b = base_shift - mm * x1
                shift = (latent_w * latent_h / (8 * 8 * 2 * 2)) * mm + b

                # Настройка ModelSamplingFlux
                sampling_base = comfy.model_sampling.ModelSamplingFlux
                sampling_type = comfy.model_sampling.CONST

                class ModelSamplingAdvanced(sampling_base, sampling_type):
                    pass

                model_sampling = ModelSamplingAdvanced(m.model.model_config)
                model_sampling.set_parameters(shift=shift)
                m.add_object_patch("model_sampling", model_sampling)

                # Настройка параметра guidance для Flux
                base_positive_with_guidance = node_helpers.conditioning_set_values(base_positive, {"guidance": guidance})

                # Используем измененную модель
                model = m
                base_positive = base_positive_with_guidance

            # Инициализируем sampler
            sampler = comfy.samplers.KSampler(
                model=model,
                steps=steps,
                device=model.load_device,
                sampler=sampler_name,
                scheduler=scheduler,
                denoise=denoise,
                model_options=model.model_options
            )

            # Подготавливаем шум
            noise = comfy.sample.prepare_noise(latent["samples"], seed)
            if noise is None:
                raise RuntimeError("Failed to generate noise")

            # Подготавливаем латентное изображение
            latent_image = latent["samples"].to(model.load_device)

            # Масштабируем маски к размеру латента
            latent_h, latent_w = latent_image.shape[2:]
            scaled_masks = []
            for mask, _, _, _, _, _ in regional_prompts:
                if mask is None:
                    continue
                if mask.shape[1:] != (latent_h, latent_w):
                    mask = torch.nn.functional.interpolate(
                        mask.unsqueeze(1),
                        size=(latent_h, latent_w),
                        mode='bilinear'
                    ).squeeze(1)
                scaled_masks.append(mask.to(model.load_device))

            # Инициализируем CFGGuider
            base_cfg_guider = RegionalCFGGuider(
                model_patcher=model,
                base_positive=base_positive,
                base_negative=base_negative,
                cfg=cfg
            )

            # Добавляем региональные промпты
            for i, (mask, positive, weight, regional_cfg, mask_threshold, region_denoise) in enumerate(regional_prompts):
                if i >= len(scaled_masks):
                    continue

                # Для Flux моделей, применяем guidance к региональным промптам
                if flux:
                    positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})

                base_cfg_guider.add_regional_prompt(
                    mask=scaled_masks[i],
                    positive=positive,
                    weight=weight,
                    regional_cfg=regional_cfg,
                    mask_threshold=mask_threshold,
                    region_denoise=region_denoise
                )

            # Получаем сигмы
            sigmas = sampler.sigmas
            if sigmas is None or len(sigmas) == 0:
                raise ValueError("Invalid sigmas generated by sampler")


            x0_output = {}
            callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

            # Выполняем сэмплирование
            sampler_obj = sampler_object(sampler_name)
            samples = base_cfg_guider.sample(
                noise=noise,
                latent_image=latent_image,
                sampler=sampler_obj,
                sigmas=sigmas,
                denoise_mask=None,
                callback=callback,
                seed=seed
            )

            # Валидируем выходные данные
            if samples is None:
                raise RuntimeError("Sampling returned no output")

            if not isinstance(samples, torch.Tensor):
                raise TypeError(f"Invalid samples type: {type(samples)}")

            # Возвращаем результат
            return {
                "samples": samples.to(comfy.model_management.intermediate_device())
            }

        except Exception as e:
            print(f"Regional sampling error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


class RegionalCFGGuider(comfy.samplers.CFGGuider):
    def __init__(self, model_patcher, base_positive, base_negative, cfg):
        super().__init__(model_patcher)
        self.base_positive = base_positive
        self.base_negative = base_negative
        self.regional_prompts = []
        self.set_conds(base_positive, base_negative)
        self.cfg = cfg

    def add_regional_prompt(self, mask, positive, weight=1.0, regional_cfg=None, mask_threshold=0.5, region_denoise=1.0):
        self.regional_prompts.append((
            mask.to(self.model_patcher.load_device),
            positive,
            weight,
            regional_cfg if regional_cfg is not None else self.cfg,
            mask_threshold,
            region_denoise
        ))

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        if not self.regional_prompts:
            return super().predict_noise(x, timestep, model_options, seed)

        # Базовые предсказания с основным CFGGuider
        base_uncond = super().predict_noise(x, timestep, {"cond": self.base_negative}, seed)
        base_cond = super().predict_noise(x, timestep, {"cond": self.base_positive}, seed)

        # Инициализируем суммарную дельту
        total_delta = torch.zeros_like(base_cond)
        combined_mask = torch.zeros_like(x[:,:1,:,:])

        # Обрабатываем каждый регион с отдельным CFGGuider
        for mask, positive, weight, regional_cfg, mask_threshold, region_denoise in self.regional_prompts:
            # Создаём отдельный CFGGuider для региона
            region_guider = comfy.samplers.CFGGuider(self.model_patcher)
            region_guider.inner_model = self.inner_model
            # Подготавливаем conds, как в set_conds
            region_guider.original_conds = {}
            region_guider.inner_set_conds({"positive": positive, "negative": self.base_negative})
            # Полностью обрабатываем conds, как в inner_sample
            region_guider.conds = {}
            for k in region_guider.original_conds:
                region_guider.conds[k] = list(map(lambda a: a.copy(), region_guider.original_conds[k]))
            region_guider.conds = comfy.samplers.process_conds(self.inner_model, x, region_guider.conds, self.model_patcher.load_device, None, None, seed)
            region_guider.set_cfg(regional_cfg)

            # Получаем предсказание для региона
            region_cond = region_guider.predict_noise(x, timestep, model_options, seed)

            # Вычисляем дельту региона относительно базового позитива
            region_delta = (region_cond - base_cond) * weight

            # Подготавливаем маску
            mask = mask.unsqueeze(1).expand_as(combined_mask)
            mask = (mask > mask_threshold).float()  # Используем индивидуальный порог
            mask = mask * region_denoise  # Масштабируем маску по уровню денойзинга региона

            # Накопление масок и дельт
            total_delta += mask * region_delta
            combined_mask = torch.clamp(combined_mask + mask, 0.0, 1.0)

        # Комбинируем с базовым CFG
        final_output = base_uncond + (base_cond - base_uncond) * self.cfg
        final_output = final_output * (1 - combined_mask) + (final_output + total_delta) * combined_mask

        return final_output


# Регистрируем ноду
RPSAMPLER_CLASS_MAPPINGS = {
    "RegionalPromptSamplerX": RegionalPromptSamplerX
}

RPSAMPLER_DISPLAY_NAME_MAPPINGS = {
    "RegionalPromptSamplerX": "Regional Prompt Sampler X"
}
