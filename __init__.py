import os
import importlib.util

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

current_dir = os.path.dirname(__file__)

for filename in os.listdir(current_dir):
    if not filename.endswith(".py") or filename == "__init__.py":
        continue

    module_name = filename[:-3]
    file_path = os.path.join(current_dir, filename)

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except ImportError:
        continue

    found_mappings = False

    for attr_name in dir(module):
        if attr_name.endswith("_CLASS_MAPPINGS"):
            class_mappings = getattr(module, attr_name, None)
            if isinstance(class_mappings, dict):
                NODE_CLASS_MAPPINGS.update(class_mappings)
                found_mappings = True

        elif attr_name.endswith("_NAME_MAPPINGS"):
            name_mappings = getattr(module, attr_name, None)
            if isinstance(name_mappings, dict):
                NODE_DISPLAY_NAME_MAPPINGS.update(name_mappings)
                found_mappings = True

    if not found_mappings:
        continue

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]
