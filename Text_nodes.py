from comfy.comfy_types import IO

class AnyType(str):
    def __eq__(self, _) -> bool:
        return True
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

class TextX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (IO.STRING, {"multiline": True}),
                "ignore_input": ("BOOLEAN", {
                    "default": True,
                    "label_on": "ignore text_input",
                    "label_off": "use text_input",
                    "forceInput": False
                }),
                "protected": ("BOOLEAN", {
                    "default": False,
                    "label_on": "protect text widget",
                    "label_off": "allow text updates",
                    "forceInput": False
                }),
            },
            "optional": {
                "text_input": (IO.STRING, {"multiline": True, "forceInput": True}),
            }
        }

    RETURN_TYPES = (any, IO.STRING,)
    RETURN_NAMES = ("TEXT", "STRING",)
    FUNCTION = "execute"
    CATEGORY = "xmtools/nodes"
    OUTPUT_NODE = True

    def execute(self, text, ignore_input=True, protected=False, text_input=None):
        output_text = text  # По умолчанию используем text

        # Если не игнорируем input и input подключен, используем его для выхода
        if not ignore_input and text_input is not None:
            output_text = text_input

            # Если защита выключена, отправляем text_input для обновления виджета text
            if not protected:
                return {"ui": {"text": (text_input,)}, "result": (output_text, output_text)}

        # Если ignore_input включен или protected включен, не обновляем виджет
        return {"result": (output_text, output_text)}



class TextConcatX:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "delimiter": (IO.STRING, {"default": "\\n"}),
            },
            "optional": {
                f"text_{i}": (IO.STRING, {"multiline": True, "forceInput": True})
                for i in range(1, 17)
            }
        }
        return inputs

    RETURN_TYPES = (any, IO.STRING,)
    RETURN_NAMES = ("TEXT", "CONCAT_TEXT",)
    FUNCTION = "concatenate"
    CATEGORY = "xmtools/nodes"

    def concatenate(self, delimiter, **kwargs):
        # Replace escaped newlines in delimiter with actual newlines
        delimiter = delimiter.replace("\\n", "\n")

        # Collect connected inputs in order
        texts = [kwargs.get(f"text_{i}", None) for i in range(1, 17)]
        connected_texts = [text for text in texts if text is not None]

        # Concatenate with delimiter
        result = delimiter.join(connected_texts)

        return (result, result,)

TEXTX_CLASS_MAPPINGS = {
    "TextX": TextX,
    "TextConcatX": TextConcatX,
}

TEXTX_NAME_MAPPINGS = {
    "TextX": "Text X",
    "TextConcatX": "Text Concat X",
}
