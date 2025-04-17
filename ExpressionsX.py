import numpy as np
import torch
import re
from numexpr import evaluate
import traceback

class ExpressionsX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "r_expression": ("STRING", {"default": "", "multiline": True}),
                "g_expression": ("STRING", {"default": "", "multiline": True}),
                "b_expression": ("STRING", {"default": "", "multiline": True}),
                "mask_expression": ("STRING", {"default": "", "multiline": True}),
                "z1": ("FLOAT", {"default": 0.5, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "z2": ("FLOAT", {"default": 0.5, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "z3": ("FLOAT", {"default": 0.5, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "z4": ("FLOAT", {"default": 0.5, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "z5": ("FLOAT", {"default": 0.5, "min": -10000.0, "max": 10000.0, "step": 0.01}),
            },
            "optional": {
                "A": ("IMAGE",),
                "B": ("IMAGE",),
                "A_mask": ("MASK",),
                "B_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "process_expression"
    CATEGORY = "xmtools/nodes"

    def clamp(self, value, min_val=0.0, max_val=1.0):
        """Ограничивает значение в заданном диапазоне"""
        return np.clip(value, min_val, max_val)

    def evaluate_expression(self, expression, context):
        """
        Оценивает выражение с использованием предоставленного контекста.
        Если выражение пустое, возвращает соответствующий канал из B.
        """
        if not expression.strip():
            # Если выражение пустое, возвращаем соответствующий канал из B
            channel = re.search(r'A\.(r|g|b|mask)', expression)
            if channel:
                ch = channel.group(1)
                if ch == 'mask':
                    return context["B_mask"]
                else:
                    return context[f"B_{ch}"]
            return context["B_r"]  # Возвращаем B.r по умолчанию если не определено

        # Безопасное выполнение выражения
        try:
            result = evaluate(expression, local_dict=context)
            return result
        except Exception as e:
            print(f"Ошибка при вычислении выражения: {expression}")
            print(f"Ошибка: {str(e)}")
            # В случае ошибки возвращаем нулевой массив той же формы
            if 'A_r' in context:
                return np.zeros_like(context['A_r'])
            else:
                return np.zeros_like(context['B_r'])

    def process_expression(self, r_expression="", g_expression="", b_expression="", mask_expression="", z1=0.5, z2=0.5, z3=0.5, z4=0.5, z5=0.5, A=None, B=None, A_mask=None, B_mask=None):
        # Check if we have at least B or B_mask
        if B is None and B_mask is None:
            raise ValueError("Error: At least B image or B_mask must be provided")

        # Determine base shape from available inputs
        if B is not None:
            B_np = B.cpu().numpy()
            base_height, base_width = B_np.shape[1:3]
            device = B.device
        else:
            B_mask_np = B_mask.cpu().numpy()
            base_height, base_width = B_mask_np.shape[1:3]
            device = B_mask.device

        base_shape = (1, base_height, base_width)
        # Create missing inputs with correct dimensions
        if B is None:
            # Create black image with base shape
            B_np = np.zeros((1, base_height, base_width, 3), dtype=np.float32)
            B = torch.from_numpy(B_np).to(device)
        else:
            B_np = B.cpu().numpy()

        if B_mask is None:
            # Create white mask with base shape
            B_mask_np = np.ones(base_shape, dtype=np.float32)
            B_mask = torch.from_numpy(B_mask_np).to(device)
        else:
            B_mask_np = B_mask.cpu().numpy()

            # Check B_mask dimensions if B exists
            if B is not None and B_mask_np.shape != B_np.shape[:-1]:
                raise ValueError(f"Error: B_mask shape {B_mask_np.shape} does not match B image shape {B_np.shape[:-1]}")

        if A is None:
            # Create black image with base shape
            A_np = np.zeros((1, base_height, base_width, 3), dtype=np.float32)
            A = torch.from_numpy(A_np).to(device)
        else:
            A_np = A.cpu().numpy()

        if A_mask is None:
            # Create white mask with base shape
            A_mask_np = np.ones(base_shape, dtype=np.float32)
            A_mask = torch.from_numpy(A_mask_np).to(device)
        else:
            A_mask_np = A_mask.cpu().numpy()

            # Check A_mask dimensions if A exists
            if A is not None and A_mask_np.shape != A_np.shape[:-1]:
                raise ValueError(f"Error: A_mask shape {A_mask_np.shape} does not match A image shape {A_np.shape[:-1]}")

        # Resize A and A_mask to match B dimensions if needed
        if A_np.shape[1:3] != B_np.shape[1:3]:
            # print(f"Image A shape {A_np.shape} differs from B shape {B_np.shape}. Resizing to match B.")

            # Get dimensions
            a_h, a_w = A_np.shape[1:3]
            b_h, b_w = B_np.shape[1:3]

            # Create new arrays filled with zeros
            new_A_np = np.zeros((A_np.shape[0], b_h, b_w, A_np.shape[3]), dtype=A_np.dtype)
            new_A_mask_np = np.zeros((A_mask_np.shape[0], b_h, b_w), dtype=A_mask_np.dtype)

            # Calculate coordinates for insertion/cropping (centering)
            start_h_a = max(0, (a_h - b_h) // 2)
            start_w_a = max(0, (a_w - b_w) // 2)
            start_h_b = max(0, (b_h - a_h) // 2)
            start_w_b = max(0, (b_w - a_w) // 2)

            # Define copy region dimensions
            copy_h = min(a_h, b_h)
            copy_w = min(a_w, b_w)

            # Copy data with centering
            new_A_np[:, start_h_b:start_h_b+copy_h, start_w_b:start_w_b+copy_w, :] = A_np[:, start_h_a:start_h_a+copy_h, start_w_a:start_w_a+copy_w, :]
            new_A_mask_np[:, start_h_b:start_h_b+copy_h, start_w_b:start_w_b+copy_w] = A_mask_np[:, start_h_a:start_h_a+copy_h, start_w_a:start_w_a+copy_w]

            # Replace original arrays with new ones
            A_np = new_A_np
            A_mask_np = new_A_mask_np

            # Convert back to PyTorch tensors
            A = torch.from_numpy(A_np).to(device)
            A_mask = torch.from_numpy(A_mask_np).to(device)

        # Create coordinate grids for x and y
        y_coords, x_coords = np.mgrid[0:base_height, 0:base_width]

        # Normalize to [0,1] range
        x_norm = x_coords.astype(np.float32) / max(base_width-1, 1)
        y_norm = y_coords.astype(np.float32) / max(base_height-1, 1)

        # Reshape to match image dimensions (add batch dimension)
        x_norm = np.broadcast_to(x_norm, base_shape)
        y_norm = np.broadcast_to(y_norm, base_shape)

        # Create context for expressions
        context = {
            "A_r": A_np[..., 0],
            "A_g": A_np[..., 1],
            "A_b": A_np[..., 2],
            "B_r": B_np[..., 0],
            "B_g": B_np[..., 1],
            "B_b": B_np[..., 2],
            "A_mask": A_mask_np,
            "B_mask": B_mask_np,
            "x": x_norm,
            "y": y_norm,
            "z1": z1,
            "z2": z2,
            "z3": z3,
            "z4": z4,
            "z5": z5,
            "min": np.minimum,
            "max": np.maximum,
            "floor": np.floor,
            "ceil": np.ceil,
            "abs": np.abs,
            "cos": np.cos,
            "sin": np.sin,
            "pow": np.power,
            "sqrt": np.sqrt,
            "clamp": self.clamp
        }

        # Process expressions for each channel
        try:
            # Red channel
            if not r_expression or not r_expression.strip():
                r_result = context["B_r"]
            else:
                result = evaluate(r_expression, local_dict=context)
                if np.isscalar(result):
                    r_result = np.full(base_shape, result, dtype=np.float32)
                else:
                    r_result = result

            # Green channel
            if not g_expression or not g_expression.strip():
                g_result = context["B_g"]
            else:
                result = evaluate(g_expression, local_dict=context)
                if np.isscalar(result):
                    g_result = np.full(base_shape, result, dtype=np.float32)
                else:
                    g_result = result

            # Blue channel
            if not b_expression or not b_expression.strip():
                b_result = context["B_b"]
            else:
                result = evaluate(b_expression, local_dict=context)
                if np.isscalar(result):
                    b_result = np.full(base_shape, result, dtype=np.float32)
                else:
                    b_result = result

            # Mask
            if not mask_expression or not mask_expression.strip():
                mask_result = context["B_mask"]
            else:
                result = evaluate(mask_expression, local_dict=context)
                if np.isscalar(result) or (isinstance(result, np.ndarray) and result.size == 1):
                    mask_result = np.full(base_shape, float(result.item() if isinstance(result, np.ndarray) else result), dtype=np.float32)
                else:
                    mask_result = np.array(result, dtype=np.float32).reshape(base_shape)

            # Ensure all channels have the same shape before stacking
            if r_result.shape != base_shape:
                r_result = np.broadcast_to(r_result, base_shape)
            if g_result.shape != base_shape:
                g_result = np.broadcast_to(g_result, base_shape)
            if b_result.shape != base_shape:
                b_result = np.broadcast_to(b_result, base_shape)

            # Assemble result image
            result_image = np.stack([r_result, g_result, b_result], axis=-1)

        except Exception as e:
            print(f"Error in processing: {str(e)}")
            traceback.print_exc()
            if B is not None:
                return (B, B_mask if B_mask is not None else A_mask)
            else:
                # Create black image as fallback
                fallback = torch.zeros((1, base_height, base_width, 3), dtype=torch.float32).to(device)
                return (fallback, B_mask if B_mask is not None else A_mask)

        # Clip values
        result_image = np.clip(result_image, 0.0, 1.0)
        mask_result = np.clip(mask_result, 0.0, 1.0)

        # Convert back to PyTorch tensors
        result_image_tensor = torch.from_numpy(result_image).to(device)
        mask_result_tensor = torch.from_numpy(mask_result).to(device)

        return (result_image_tensor, mask_result_tensor,)

EXPRESIIONSX_CLASS_MAPPINGS = {
    "ExpressionsX": ExpressionsX
}

EXPRESIIONSX_NAME_MAPPINGS = {
    "ExpressionsX": "Expressions X"
}
