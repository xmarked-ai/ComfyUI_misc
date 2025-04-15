import torch
from transformers import BlipProcessor, BlipForImageTextRetrieval, BlipForConditionalGeneration, BlipForQuestionAnswering
from PIL import Image

# Глобальные переменные
BLIP_ITM_MODEL = None
BLIP_ITM_PROCESSOR = None
BLIP_CAP_MODEL = None
BLIP_CAP_PROCESSOR = None
BLIP_VQA_MODEL = None
BLIP_VQA_PROCESSOR = None
DEVICE = None
CURRENT_MODEL = None  # "itm", "cap", "vqa" или None

class BLIPMatcherX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"multiline": True}),
                "mode": ("BOOLEAN", {"default": True, "label_on": "match score", "label_off": "text extraction"}),
                "type": ("BOOLEAN", {"default": True, "label_on": "captioning", "label_off": "vqa"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "match"
    CATEGORY = "xmtools/nodes"

    def match(self, image, text, mode, type):
        global BLIP_ITM_MODEL, BLIP_ITM_PROCESSOR, BLIP_CAP_MODEL, BLIP_CAP_PROCESSOR, BLIP_VQA_MODEL, BLIP_VQA_PROCESSOR, DEVICE, CURRENT_MODEL

        # Устанавливаем устройство
        if DEVICE is None:
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Преобразуем ComfyUI-тензор в PIL Image
        image_tensor = image[0]
        image_np = image_tensor.cpu().numpy() * 255
        image_np = image_np.astype("uint8")
        image_pil = Image.fromarray(image_np).convert("RGB")

        def unload_model():
            """Выгружает текущую модель из памяти"""
            global BLIP_ITM_MODEL, BLIP_ITM_PROCESSOR, BLIP_CAP_MODEL, BLIP_CAP_PROCESSOR, BLIP_VQA_MODEL, BLIP_VQA_PROCESSOR, CURRENT_MODEL
            if CURRENT_MODEL == "itm":
                del BLIP_ITM_MODEL
                del BLIP_ITM_PROCESSOR
                BLIP_ITM_MODEL = None
                BLIP_ITM_PROCESSOR = None
            elif CURRENT_MODEL == "cap":
                del BLIP_CAP_MODEL
                del BLIP_CAP_PROCESSOR
                BLIP_CAP_MODEL = None
                BLIP_CAP_PROCESSOR = None
            elif CURRENT_MODEL == "vqa":
                del BLIP_VQA_MODEL
                del BLIP_VQA_PROCESSOR
                BLIP_VQA_MODEL = None
                BLIP_VQA_PROCESSOR = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            CURRENT_MODEL = None

        if mode:  # Mode=True: ITM
            if CURRENT_MODEL != "itm":
                if CURRENT_MODEL is not None:
                    unload_model()
                BLIP_ITM_PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
                BLIP_ITM_MODEL = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to(DEVICE)
                BLIP_ITM_MODEL.eval()
                CURRENT_MODEL = "itm"

            inputs = BLIP_ITM_PROCESSOR(images=image_pil, text=text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = BLIP_ITM_MODEL(**inputs)
                prob = torch.softmax(outputs.itm_score, dim=-1)[0, 1].item()
            result = f"{prob:.3f}"

        else:  # Mode=False: экстракция текста
            if type:  # Type=True: Captioning
                if CURRENT_MODEL != "cap":
                    if CURRENT_MODEL is not None:
                        unload_model()
                    BLIP_CAP_PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
                    BLIP_CAP_MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(DEVICE)
                    BLIP_CAP_MODEL.eval()
                    CURRENT_MODEL = "cap"

                inputs = BLIP_CAP_PROCESSOR(images=image_pil, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    out = BLIP_CAP_MODEL.generate(**inputs)
                caption = BLIP_CAP_PROCESSOR.batch_decode(out, skip_special_tokens=True)[0]
                text_words = [word.strip() for word in text.split(",")]
                matching_words = [word for word in text_words if word.lower() in caption.lower()]
                result = ", ".join(matching_words) if matching_words else "none"

            else:  # Type=False: VQA
                if CURRENT_MODEL != "vqa":
                    if CURRENT_MODEL is not None:
                        unload_model()
                    BLIP_VQA_PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
                    BLIP_VQA_MODEL = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large").to(DEVICE)
                    BLIP_VQA_MODEL.eval()
                    CURRENT_MODEL = "vqa"

                text_words = [word.strip() for word in text.split(",")]
                matching_words = []
                for word in text_words:
                    question = f"Is there a {word} in the image?"
                    inputs = BLIP_VQA_PROCESSOR(images=image_pil, text=question, return_tensors="pt").to(DEVICE)
                    with torch.no_grad():
                        out = BLIP_VQA_MODEL.generate(**inputs)
                    answer = BLIP_VQA_PROCESSOR.batch_decode(out, skip_special_tokens=True)[0].lower()
                    if answer in ["yes", "yep", "yeah"]:
                        matching_words.append(word)
                result = ", ".join(matching_words) if matching_words else "none"

        return (result,)

BLIPMATCHER_CLASS_MAPPINGS = {
    "BLIPMatcherX": BLIPMatcherX
}

BLIPMATCHER_NAME_MAPPINGS = {
    "BLIPMatcherX": "BLIP Matcher X"
}