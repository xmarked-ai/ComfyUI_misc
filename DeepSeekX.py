from openai import OpenAI

# Объявление any_type
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

class DeepSeekX:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "system_prompt": ("STRING", {"default": "Переведи текст с русского на английский", "multiline": True}),
                "user_prompt": ("STRING", {"multiline": True}),
                "max_tokens": ("INT", {"default": 1024, "min": 20, "max": 4096}),
            }
        }

    RETURN_TYPES = (any_type,)  # Используем any_type
    RETURN_NAMES = ("output",)
    CATEGORY = "xmtools/nodes"
    FUNCTION = "execute"

    def execute(self, api_key, system_prompt, user_prompt, max_tokens):
        # Инициализация клиента OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        # Отправка запроса к DeepSeek API
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
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

DEEPSEEKX_CLASS_MAPPINGS = {
    "DeepSeekX": DeepSeekX,
}

DEEPSEEKX_DISPLAY_NAME_MAPPINGS = {
    "DeepSeekX": "DeepSeek X",
}
