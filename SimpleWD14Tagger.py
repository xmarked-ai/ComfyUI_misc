import os
import numpy as np
import csv
import onnxruntime as ort
from PIL import Image
import comfy.utils
import folder_paths
import requests
import tqdm

# Константы по умолчанию
DEFAULT_MODEL = "wd-v1-4-moat-tagger-v2"
DEFAULT_THRESHOLD = 0.35
DEFAULT_CHARACTER_THRESHOLD = 0.85
DEFAULT_REPLACE_UNDERSCORE = False
DEFAULT_TRAILING_COMMA = False
DEFAULT_EXCLUDE_TAGS = ""
HF_ENDPOINT = "https://huggingface.co"

# Список доступных моделей
MODELS = {
    "wd-eva02-large-tagger-v3": f"{HF_ENDPOINT}/SmilingWolf/wd-eva02-large-tagger-v3",
    "wd-vit-tagger-v3": f"{HF_ENDPOINT}/SmilingWolf/wd-vit-tagger-v3",
    "wd-swinv2-tagger-v3": f"{HF_ENDPOINT}/SmilingWolf/wd-swinv2-tagger-v3",
    "wd-convnext-tagger-v3": f"{HF_ENDPOINT}/SmilingWolf/wd-convnext-tagger-v3",
    "wd-v1-4-moat-tagger-v2": f"{HF_ENDPOINT}/SmilingWolf/wd-v1-4-moat-tagger-v2",
    "wd-v1-4-convnextv2-tagger-v2": f"{HF_ENDPOINT}/SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
    "wd-v1-4-convnext-tagger-v2": f"{HF_ENDPOINT}/SmilingWolf/wd-v1-4-convnext-tagger-v2",
    "wd-v1-4-convnext-tagger": f"{HF_ENDPOINT}/SmilingWolf/wd-v1-4-convnext-tagger",
    "wd-v1-4-vit-tagger-v2": f"{HF_ENDPOINT}/SmilingWolf/wd-v1-4-vit-tagger-v2",
    "wd-v1-4-swinv2-tagger-v2": f"{HF_ENDPOINT}/SmilingWolf/wd-v1-4-swinv2-tagger-v2",
    "wd-v1-4-vit-tagger": f"{HF_ENDPOINT}/SmilingWolf/wd-v1-4-vit-tagger",
    "Z3D-E621-Convnext": f"{HF_ENDPOINT}/silveroxides/Z3D-E621-Convnext"
}

# Настройка путей и директории для моделей
def setup_models_dir():
    if "wd14_tagger" in folder_paths.folder_names_and_paths:
        models_dir = folder_paths.get_folder_paths("wd14_tagger")[0]
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
    else:
        models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
    return models_dir

# Получение списка установленных моделей
def get_installed_models(models_dir):
    models = []
    if os.path.exists(models_dir):
        files = os.listdir(models_dir)
        onnx_files = [f for f in files if f.endswith(".onnx")]
        for onnx_file in onnx_files:
            base_name = os.path.splitext(onnx_file)[0]
            if os.path.exists(os.path.join(models_dir, base_name + ".csv")):
                models.append(onnx_file)
    return models

# Функция для скачивания файла с прогресс-баром
def download_file(url, destination):
    print(f"Скачивание: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192

    with open(destination, 'wb') as f:
        with tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(destination)) as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"Скачивание завершено: {destination}")

# Функция для скачивания модели и CSV
def download_model(model_name, models_dir):
    if model_name not in MODELS:
        print(f"Модель {model_name} не найдена в списке доступных моделей")
        return False

    base_url = MODELS[model_name]
    model_url = f"{base_url}/resolve/main/model.onnx"
    csv_url = f"{base_url}/resolve/main/selected_tags.csv"

    model_path = os.path.join(models_dir, f"{model_name}.onnx")
    csv_path = os.path.join(models_dir, f"{model_name}.csv")

    try:
        print(f"Скачивание модели {model_name}...")
        download_file(model_url, model_path)
        print(f"Скачивание тегов для {model_name}...")
        download_file(csv_url, csv_path)
        return True
    except Exception as e:
        print(f"Ошибка при скачивании модели {model_name}: {str(e)}")
        # Удаляем частично скачанные файлы
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(csv_path):
            os.remove(csv_path)
        return False

# Основная функция тэггирования
def tag_image(image, model_path, csv_path, threshold=DEFAULT_THRESHOLD,
             character_threshold=DEFAULT_CHARACTER_THRESHOLD,
             exclude_tags=DEFAULT_EXCLUDE_TAGS,
             replace_underscore=DEFAULT_REPLACE_UNDERSCORE,
             trailing_comma=DEFAULT_TRAILING_COMMA):
    # Доступные провайдеры для ONNXRuntime
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # Загрузка модели
    model = ort.InferenceSession(model_path, providers=providers)

    # Получаем информацию о входном тензоре
    input = model.get_inputs()[0]
    height = input.shape[1]

    # Подготовка изображения
    ratio = float(height)/max(image.size)
    new_size = tuple([int(x*ratio) for x in image.size])
    image = image.resize(new_size, Image.LANCZOS)
    square = Image.new("RGB", (height, height), (255, 255, 255))
    square.paste(image, ((height-new_size[0])//2, (height-new_size[1])//2))

    # Преобразование в формат для модели
    image = np.array(square).astype(np.float32)
    image = image[:, :, ::-1]  # RGB -> BGR
    image = np.expand_dims(image, 0)

    # Загрузка тегов из CSV
    tags = []
    general_index = None
    character_index = None

    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # Пропускаем заголовок
        for row in reader:
            if general_index is None and row[2] == "0":
                general_index = reader.line_num - 2
            elif character_index is None and row[2] == "4":
                character_index = reader.line_num - 2

            if replace_underscore:
                tags.append(row[1].replace("_", " "))
            else:
                tags.append(row[1])

    # Запуск инференса
    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input.name: image})[0]

    # Обработка результатов
    result = list(zip(tags, probs[0]))

    # Фильтрация тегов
    general = [item for item in result[general_index:character_index] if item[1] > threshold]
    character = [item for item in result[character_index:] if item[1] > character_threshold]

    # Объединение результатов
    all_tags = character + general

    # Удаление исключенных тегов
    if exclude_tags:
        # Заменяем переносы строк на запятые и нормализуем пробелы вокруг запятых
        exclude_tags = exclude_tags.replace('\n', ',')
        remove = [s.strip() for s in exclude_tags.lower().split(",") if s.strip()]
        all_tags = [tag for tag in all_tags if tag[0].lower() not in remove]

    # Форматирование результата
    if trailing_comma:
        tags_text = "".join((item[0].replace("(", "\\(").replace(")", "\\)") + ", " for item in all_tags))
    else:
        tags_text = ", ".join((item[0].replace("(", "\\(").replace(")", "\\)") for item in all_tags))

    return tags_text

# Класс ноды для ComfyUI
class SimpleWD14TaggerX:
    @classmethod
    def INPUT_TYPES(s):
        # Инициализация списка моделей
        models_dir = setup_models_dir()
        installed_models = [os.path.splitext(m)[0] for m in get_installed_models(models_dir)]

        # Объединяем установленные модели и доступные модели из списка
        all_models = list(MODELS.keys())
        for model in installed_models:
            if model not in all_models:
                all_models.append(model)

        return {"required": {
            "image": ("IMAGE", ),
            "model": (all_models, {"default": DEFAULT_MODEL}),
            "threshold": ("FLOAT", {"default": DEFAULT_THRESHOLD, "min": 0.0, "max": 1.0, "step": 0.05}),
            "character_threshold": ("FLOAT", {"default": DEFAULT_CHARACTER_THRESHOLD, "min": 0.0, "max": 1.0, "step": 0.05}),
            "replace_underscore": ("BOOLEAN", {"default": DEFAULT_REPLACE_UNDERSCORE}),
            "trailing_comma": ("BOOLEAN", {"default": DEFAULT_TRAILING_COMMA}),
            "exclude_tags": ("STRING", {"default": DEFAULT_EXCLUDE_TAGS, "multiline": True}),
        }}

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "tag_images"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def tag_images(self, image, model, threshold, character_threshold, exclude_tags="", replace_underscore=False, trailing_comma=False):
        models_dir = setup_models_dir()
        model_path = os.path.join(models_dir, model + ".onnx")
        csv_path = os.path.join(models_dir, model + ".csv")

        # Проверяем, существуют ли файлы модели, если нет - скачиваем
        if not (os.path.exists(model_path) and os.path.exists(csv_path)):
            print(f"Модель {model} не найдена локально. Пытаюсь скачать...")
            success = download_model(model, models_dir)
            if not success:
                return {"ui": {"tags": [f"Ошибка: Не удалось скачать модель {model}"]},
                        "result": ([f"Ошибка: Не удалось скачать модель {model}"],)}

        # Обработка пакета изображений
        tensor = image * 255
        tensor = np.array(tensor, dtype=np.uint8)

        pbar = comfy.utils.ProgressBar(tensor.shape[0])
        tags = []

        for i in range(tensor.shape[0]):
            img = Image.fromarray(tensor[i])
            tags_text = tag_image(
                img, model_path, csv_path, threshold, character_threshold,
                exclude_tags, replace_underscore, trailing_comma
            )
            tags.append(tags_text)
            pbar.update(1)

        return {"ui": {"tags": tags}, "result": (tags,)}

# Регистрация ноды
NODE_CLASS_MAPPINGS = {
    "SimpleWD14TaggerX": SimpleWD14TaggerX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleWD14TaggerX": "Simple WD14 Tagger X",
}
