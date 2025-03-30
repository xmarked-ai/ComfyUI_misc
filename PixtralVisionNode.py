import json
import requests
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from nodes import MAX_RESOLUTION
from io import BytesIO

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

class PixtralVisionX:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "server_address": ("STRING", {"default": "192.168.98.205:5001"}),
                "system_prompt": ("STRING", {"default": "You are a helpful AI assistant that accurately describes images.", "multiline": True}),
                "query": ("STRING", {"default": "Describe this image in 30 words", "multiline": True}),
                "max_tokens": ("INT", {"default": 1024, "min": 5, "max": 4096}),
            },
        }

    RETURN_TYPES = (any_type, "STRING", )
    RETURN_NAMES = ("descriptions", "strings",)
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "process_image"
    CATEGORY = "xmtools/nodes"

    FORCE_EXECUTION = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        import random
        return random.random()

    def process_image(self, image, server_address, system_prompt, query, max_tokens):
        try:
            descriptions = []
            batch_size = image.shape[0]  # Количество изображений в батче

            for i in range(batch_size):
                # Преобразуем изображение в PIL
                image_tensor = image[i]
                image_array = (image_tensor * 255).byte().cpu().numpy()
                image_pil = Image.fromarray(image_array, mode="RGB")

                # Конвертируем в байты
                img_byte_arr = BytesIO()
                image_pil.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)

                # Отправляем запрос на сервер
                url = f"http://{server_address}/process_image"
                files = {'image': ('image.png', img_byte_arr, 'image/png')}
                data = {
                    'query': query,
                    'system_prompt': system_prompt,
                    'max_tokens': str(max_tokens),  # Form-data требует строки
                    'stream': 'false'  # Без стрима
                }

                response = requests.post(url, files=files, data=data)
                response.raise_for_status()

                data = response.json()
                if 'error' in data:
                    description = f"Ошибка: {data['error']}"
                else:
                    description = data['text']

                descriptions.append(description)

            return (descriptions, "\n".join(descriptions))

        except Exception as e:
            return ([f"Ошибка при обработке изображения: {str(e)}"], f"Ошибка при обработке изображения: {str(e)}")

PIXTRALVISIONX_CLASS_MAPPINGS = {
    "PixtralVisionX": PixtralVisionX,
}

PIXTRALVISIONX_NAME_MAPPINGS = {
    "PixtralVisionX": "Pixtral Vision X",
}
