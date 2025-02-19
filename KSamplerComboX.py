import os

import torch
import numpy
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import math

import comfy.sd
import comfy.utils
import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy_extras.nodes_custom_sampler
import latent_preview
import folder_paths
import node_helpers

from comfy.comfy_types import IO
import nodes

MAX_RESOLUTION=16384

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

class KSamplerComboX:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
                "vae": ("VAE", {"tooltip": "The VAE model used for decoding the latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple", "tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
                "flux": ("BOOLEAN", {"default": True, "label_on": "flux", "label_off": "sd15 sdxl flux", "forceInput": False}),
                "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 100.0, "step":0.01}),
                "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step":0.01}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "width": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "positive": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "negative": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "ksampler_combo_x"

    CATEGORY = "loaders"

    def ksampler_combo_x(self, model, clip, vae, seed, steps, cfg, sampler_name, scheduler, denoise=1.0, flux=True, max_shift=1.15, base_shift=0.5, guidance=3.5, width=1024, height=1024 ,positive="", negative="", lora_name="", strength_model=1.0, strength_clip=0.0):

        latent_image = {"samples":torch.zeros([1, 16, height // 8  , width // 8], device=comfy.model_management.intermediate_device())}

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

        tokens   = clip_lora.tokenize(positive)
        cond_pos = clip_lora.encode_from_tokens_scheduled(tokens)
        tokens   = clip_lora.tokenize(negative)
        cond_neg = clip_lora.encode_from_tokens_scheduled(tokens)


        if flux:
            ############################## if flux #############################

            #### Model Sampling Flux
            m = model_lora.clone()

            x1 = 256
            x2 = 4096
            mm = (max_shift - base_shift) / (x2 - x1)
            b = base_shift - mm * x1
            shift = (width * height / (8 * 8 * 2 * 2)) * mm + b

            sampling_base = comfy.model_sampling.ModelSamplingFlux
            sampling_type = comfy.model_sampling.CONST

            class ModelSamplingAdvanced(sampling_base, sampling_type):
                pass

            model_sampling = ModelSamplingAdvanced(model_lora.model.model_config)
            model_sampling.set_parameters(shift=shift)
            m.add_object_patch("model_sampling", model_sampling)

            #### Flux Guidance
            c = node_helpers.conditioning_set_values(cond_pos, {"guidance": guidance})

            #### BasicGuider
            guider = comfy_extras.nodes_custom_sampler.Guider_Basic(m)
            guider.set_conds(c)

            #### BasicScheduler
            total_steps = steps
            if denoise < 1.0:
                if denoise <= 0.0:
                    return (torch.FloatTensor([]),)
                total_steps = int(steps/denoise)

            sigmas = comfy.samplers.calculate_sigmas(m.get_model_object("model_sampling"), scheduler, total_steps).cpu()
            sigmas = sigmas[-(steps + 1):]

            #### Ksampler Select
            sampler = comfy.samplers.sampler_object(sampler_name)

            #### Random Noise
            rand_noise = comfy_extras.nodes_custom_sampler.Noise_RandomNoise(seed)

            #### Sampler Custom Advanced
            latent = latent_image.copy()
            latent_img = latent["samples"]
            latent = latent.copy()
            latent_img = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_img)
            latent["samples"] = latent_img

            noise_mask = None
            if "noise_mask" in latent:
                noise_mask = latent["noise_mask"]

            x0_output = {}
            callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
            samples = guider.sample(rand_noise.generate_noise(latent), latent_img, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=rand_noise.seed)
            samples = samples.to(comfy.model_management.intermediate_device())

            decoded_image = vae.decode(samples)

        else:
            sampled_samples = nodes.common_ksampler(model_lora, seed, steps, cfg, sampler_name, scheduler, cond_pos, cond_neg, latent_image, denoise=denoise)
            decoded_image = vae.decode(sampled_samples[0]["samples"])

        return (decoded_image,)

KSAMPLERCOMBOX_CLASS_MAPPINGS = {
    "KSamplerComboX": KSamplerComboX,
}

KSAMPLERCOMBOX_NAME_MAPPINGS = {
    "KSamplerComboX": "KSampler Combo X",
}
