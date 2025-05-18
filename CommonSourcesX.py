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

class CommonSourcesX:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # "positive": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                # "negative": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 6.5, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple", "tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "width": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
            },
            # "optional": {
            #     "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
            # },
        }

    # RETURN_TYPES = (IO.CONDITIONING, IO.CONDITIONING, "INT", "INT", "FLOAT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "LATENT", "INT", "INT",)
    # RETURN_NAMES = ("COND POSIVIVE", "COND NEGATIVE", "SEED", "STEPS", "CFG", "SAMPLER", "SCHEDULER", "LATENT", "WIDTH", "HEIGHT", )
    RETURN_TYPES = ("INT", "INT", "FLOAT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "LATENT", "INT", "INT",)
    RETURN_NAMES = ("seed", "steps", "cfg", "sampler", "scheduler", "latent", "width", "height",)
    FUNCTION = "common_sources"
    CATEGORY = "xmtools/nodes"

    # def common_sources(self, positive, negative, seed=None, steps=None, cfg=None, sampler_name=None, scheduler=None, width=None, height=None, clip=None):
    def common_sources(self, seed=None, steps=None, cfg=None, sampler_name=None, scheduler=None, width=None, height=None, clip=None):

        latent_image = {"samples":torch.zeros([1, 16, height // 8  , width // 8], device=comfy.model_management.intermediate_device())}

        # tokens   = clip.tokenize(positive)
        # cond_pos = clip.encode_from_tokens_scheduled(tokens)

        # tokens   = clip.tokenize(negative)
        # cond_neg = clip.encode_from_tokens_scheduled(tokens)

        # return (cond_pos, cond_neg, seed, steps, cfg, sampler_name, scheduler, latent_image, width, height,)
        return (seed, steps, cfg, sampler_name, scheduler, latent_image, width, height,)

NODE_CLASS_MAPPINGS = {
    "CommonSourcesX": CommonSourcesX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CommonSourcesX": "Common Sources X",
}
