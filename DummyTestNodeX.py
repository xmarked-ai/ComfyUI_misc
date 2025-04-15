class DummyTestNodeX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "dummy_master": ("BOOLEAN", {
                    "default": True,
                    "label_on": "master",
                    "label_off": "channels"
                }),
                "dummy": ("FLOAT", {"default": 1.0, "step": 0.01}),
            },
            "optional": {
                "dummy_r": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "dummy_g": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "dummy_b": ("FLOAT", {"default": 1.0, "step": 0.01}),
            },
            "visibility": {
                # если dummy_master включен (True), то скрыть dummy_r/g/b
                "dummy_r": "!dummy_master",
                "dummy_g": "!dummy_master",
                "dummy_b": "!dummy_master",
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "passthrough"
    CATEGORY = "xmtools/nodes"

    def passthrough(self, image, dummy_master, dummy, dummy_r=1.0, dummy_g=1.0, dummy_b=1.0):
        # никакой обработки, просто возврат изображения
        return (image,)


NODE_CLASS_MAPPINGS = {
    "DummyTestNodeX": DummyTestNodeX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DummyTestNodeX": "Dummy Test Node X",
}
