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
                "system_prompt": ("STRING", {"default": "You are a helpful AI assistant that accurately describes images.", "multiline": True}),  # Многострочное поле для запроса
                "query": ("STRING", {"default": "Describe this image in 30 words", "multiline": True}),  # Многострочное поле для запроса
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
        # Всегда возвращаем новое значение, заставляя ComfyUI думать, что нода изменилась
        import random
        return random.random()

    def process_image(self, image, server_address, system_prompt, query, max_tokens):
        try:
            descriptions = []
            batch_size = image.shape[0]  # Получаем количество изображений в batch

            for i in range(batch_size):
                # Преобразуем изображение в формат PIL
                image_tensor = image[i]  # Берем i-ое изображение из batch
                image_array = (image_tensor * 255).byte().cpu().numpy()  # Преобразуем в numpy массив
                image_pil = Image.fromarray(image_array, mode="RGB")  # Создаем PIL изображение

                # Конвертируем PIL изображение в байты в памяти
                img_byte_arr = BytesIO()
                image_pil.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)  # Возвращаемся в начало потока

                # Отправляем запрос на сервер
                url = f"http://{server_address}/process_image_stream"
                files = {'image': ('image.png', img_byte_arr, 'image/png')}
                data = {'query': query, 'system_prompt': system_prompt, 'max_tokens': max_tokens}

                response = requests.post(url, files=files, data=data, stream=True)
                description = ""

                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])
                                if 'error' in data:
                                    description = f"Ошибка: {data['error']}"
                                else:
                                    description = data['text']
                            except json.JSONDecodeError as e:
                                description = f"Ошибка декодирования JSON: {str(e)}"

                descriptions.append(description)


            # Возвращаем список описаний
            return (descriptions, "\n".join(descriptions))

        except Exception as e:
            return ([f"Ошибка при обработке изображения: {str(e)}"], f"Ошибка при обработке изображения: {str(e)}")

PIXTRALVISIONX_CLASS_MAPPINGS = {
    "PixtralVisionX": PixtralVisionX,
}

PIXTRALVISIONX_NAME_MAPPINGS = {
    "PixtralVisionX": "Pixtral Vision X",
}
