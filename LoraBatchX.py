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
import latent_preview
import folder_paths

from comfy.comfy_types import IO
import nodes

MAX_RESOLUTION=16384

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

class LoraBatchSamplerX:

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
                "width": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "positive": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "negative": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "loras_dir": ("STRING", {"default": (os.path.normpath(os.path.join(folder_paths.models_dir, "loras")))}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
                "images_per_row": ("INT", {"default": 3, "min": 0, "max": 7, "tooltip": "The number of images per row."}),
                "padding": ("INT", {"default": 10, "min": 0, "max": 100}),
                "print_lora_name": ("BOOLEAN", {"default": True, "label_on": "print", "label_off": "do not print", "forceInput": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "batch_sampler_x"

    CATEGORY = "loaders"

    def batch_sampler_x(self, model, clip, vae, seed, steps, cfg, sampler_name, scheduler, denoise=1.0, width=1024, height=1024 ,positive="", negative="", loras_dir="", strength_model=1.0, strength_clip=0.0, images_per_row=3, padding=10, print_lora_name=True):
        names = [f for f in os.listdir(loras_dir) if os.path.isfile(os.path.join(loras_dir,f))]
        names = sorted(names, key=str.casefold)
        label_height = 50

        tokens   = clip.tokenize(positive)
        cond_pos = clip.encode_from_tokens_scheduled(tokens)
        tokens   = clip.tokenize(negative)
        cond_neg = clip.encode_from_tokens_scheduled(tokens)

        latent_image = {"samples":torch.zeros([1, 16, height // 8  , width // 8], device=comfy.model_management.intermediate_device())}

        images = []
        generated_images = []

        for name in names:
            if not name.endswith(('.pt', '.safetensors')):
                continue

            lora_path = os.path.normpath(os.path.join(loras_dir, name))
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

            sampled_samples = nodes.common_ksampler(model_lora, seed, steps, cfg, sampler_name, scheduler, cond_pos, cond_neg, latent_image, denoise=denoise)
            decoded_image = vae.decode(sampled_samples[0]["samples"])

            if not print_lora_name:
                images.append(decoded_image)
            else:
                decoded_image = decoded_image.squeeze(0)
                pil_image = Image.fromarray((decoded_image.cpu().numpy() * 255).astype('uint8'))

                img_with_label = Image.new('RGB', (pil_image.width, pil_image.height + label_height), 'white')
                img_with_label.paste(pil_image, (0, 0))

                draw = ImageDraw.Draw(img_with_label)

                font_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "font/NotoSans-Regular.ttf"))
                font = ImageFont.truetype(font_path, size=24)

                text = os.path.splitext(os.path.basename(name))[0] + "     Sampler: " + sampler_name + "   Scheduler: " + scheduler + "   Steps: " + str(steps)
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (pil_image.width - text_width) // 2

                draw.text((x, pil_image.height + 5), text, fill='black', font=font)

                decoded_image = torch.tensor(numpy.array(img_with_label) / 255.0, dtype=torch.float32).unsqueeze(0)

            generated_images.append(decoded_image)

            del model_lora
            del clip_lora
            torch.cuda.empty_cache()

        if images_per_row == 0:
            return (generated_images,)

        n_images = len(generated_images)
        rows = math.ceil(n_images / images_per_row)

        _, height, width, channels = generated_images[0].shape

        grid = torch.zeros((1,
                           rows * height + (rows - 1) * padding,
                           min(n_images, images_per_row) * width + (min(n_images, images_per_row) - 1) * padding,
                           channels))

        for idx, img in enumerate(generated_images):
            row = idx // images_per_row
            col = idx % images_per_row
            grid[:,
                 row*(height + padding):row*(height + padding) + height,
                 col*(width + padding):col*(width + padding) + width,
                 :] = img

        return ([grid],)

LORAX_CLASS_MAPPINGS = {
    "LoraBatchSamplerX": LoraBatchSamplerX,
}

LORAX_NAME_MAPPINGS = {
    "LoraBatchSamplerX": "Lora Batch Sampler X",
}
