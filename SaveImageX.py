from __future__ import annotations
import torch
import os
import sys
import json
import datetime
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
import numpy as np
import random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
from comfy.cli_args import args
import folder_paths

class SaveImageX:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Изображения для сохранения."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "subdirectory": ("STRING", {"default": "%date%"}),
                "compress_level": ("INT", {"default": 4, "min": 0, "max": 9, "step": 1}),
                "save_file": ("BOOLEAN", {"default": True, "label_on": "save", "label_off": "preview_only", "forceInput": False}),
                "show_preview": ("BOOLEAN", {"default": True, "forceInput": False})
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "xmtools/nodes"

    def save_images(self, images, filename_prefix="ComfyUI", subdirectory="", compress_level=4, show_preview=True, save_file=True, prompt=None, extra_pnginfo=None):
        # Временно меняем настройки если save_file=False
        if not save_file:
            # Сохраняем временные файлы (как в PreviewImage)
            original_output_dir = self.output_dir
            original_type = self.type
            original_prefix_append = self.prefix_append

            self.output_dir = folder_paths.get_temp_directory()
            self.type = "temp"
            self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
            show_preview = True  # Принудительно показываем превью

        try:
            if subdirectory and "%date%" in subdirectory:
                current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                subdirectory = subdirectory.replace("%date%", current_date)

            full_output_dir = self.output_dir
            if subdirectory and save_file:
                full_output_dir = os.path.join(full_output_dir, subdirectory)
                os.makedirs(full_output_dir, exist_ok=True)

            original_filename_prefix = filename_prefix

            current_time = datetime.datetime.now()
            time_str = current_time.strftime("%Y_%m_%d_%H-%M-%S-%f")[:-3]
            filename_prefix = f"{filename_prefix}_{time_str}"
            filename_prefix += self.prefix_append

            results = []
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix, full_output_dir, images[0].shape[1], images[0].shape[0]
            )

            for batch_number, image in enumerate(images):
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

                metadata = None
                if not args.disable_metadata:
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
                file = f"{filename_with_batch_num}_{batch_number:03d}.png"
                full_path = os.path.join(full_output_folder, file)

                img.save(full_path, pnginfo=metadata, compress_level=compress_level)

                results.append({
                    "filename": file,
                    "subfolder": os.path.join(subdirectory, subfolder) if save_file else subfolder,  # Учитываем поддиректорию
                    "type": self.type
                })

            return {
                "ui": {
                    "images": results if show_preview else []  # Управление отображением превью
                },
                "result": (images,)
            }
        finally:
            if not save_file:
                # Восстанавливаем оригинальные значения
                self.output_dir = original_output_dir
                self.type = original_type
                self.prefix_append = original_prefix_append

SAVEIMAGEX_CLASS_MAPPINGS = {
    "SaveImageX": SaveImageX,
}

SAVEIMAGEX_NAME_MAPPINGS = {
    "SaveImageX": "Save Image X",
}
