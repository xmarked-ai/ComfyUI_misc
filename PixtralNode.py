from openai import OpenAI

# Объявление any_type
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

class PixtralX:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "0123456789"}),
                "server_address": ("STRING", {"default": "192.168.98.205:5001"}),
                "system_prompt": ("STRING", {"default": "Переведи текст с русского на английский", "multiline": True}),
                "user_prompt": ("STRING", {"multiline": True}),
                "max_tokens": ("INT", {"default": 1024, "min": 20, "max": 4096}),
            }
        }

    RETURN_TYPES = (any_type,)  # Используем any_type
    RETURN_NAMES = ("output",)
    CATEGORY = "xmtools/nodes"
    FUNCTION = "execute"

    def execute(self, api_key, server_address, system_prompt, user_prompt, max_tokens):
        # Инициализация клиента OpenAI
        url = f"http://{server_address}/v1"
        client = OpenAI(api_key=api_key, base_url=url)

        # Отправка запроса к DeepSeek API
        try:
            response = client.chat.completions.create(
                model="pixtral-12b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                stream=False
            )
            result = response.choices[0].message.content
        except Exception as e:
            result = f"Ошибка: {str(e)}"

        return (result,)

PIXTRALX_CLASS_MAPPINGS = {
    "PixtralX": PixtralX,
}

PIXTRALX_NAME_MAPPINGS = {
    "PixtralX": "Pixtral X",
}
