from comfy_execution.graph import ExecutionBlocker

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

class SimpleBlockerX:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (any_type, {"default": 0}),
                "block": ("BOOLEAN", {"default": True, "label_on": "block", "label_off": "pass"}),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"

    CATEGORY = "xmtools/nodes"

    def execute(self, input, block):
        if block:
            return (ExecutionBlocker(None),)

        return (input,)

BLOCKER_CLASS_MAPPINGS = {
    "SimpleBlockerX": SimpleBlockerX,
}

BLOCKER_DISPLAY_NAME_MAPPINGS = {
    "SimpleBlockerX": "Simple Blocker X",
}
