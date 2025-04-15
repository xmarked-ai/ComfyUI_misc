import numpy as np
import torch
import cv2
from scipy.ndimage import label
import math

class FourCornerPinMaskX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),          # Mask with 4 points (1, Y, X), float
                "foreground": ("IMAGE",),   # Image to distort (1, Y', X', 3)
                "background": ("IMAGE",),   # Background (1, Y, X, 3)
                "rotate": (["0", "+90", "+180", "+270"], {"default": "0"}),
                "inset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "pin_corners"
    CATEGORY = "xmtools/nodes"

    def pin_corners(self, mask, foreground, background, rotate, inset):
        # Convert tensors to numpy
        mask_np = mask.squeeze(0).cpu().numpy()  # (Y, X)
        fg_np = foreground.squeeze(0).cpu().numpy()  # (Y', X', 3)
        bg_np = background.squeeze(0).cpu().numpy()  # (Y, X, 3)

        # Binarize mask: everything non-zero becomes 1
        mask_np = (mask_np != 0).astype(np.uint8)

        # Find separate regions (points) in the mask
        labeled, num_features = label(mask_np)
        if num_features != 4:
            raise ValueError(f"Expected 4 points in mask, found {num_features}")

        # Find centroids of each region
        centroids = []
        for i in range(1, num_features + 1):  # Labels from 1 to num_features
            coords = np.where(labeled == i)
            centroid_y = np.mean(coords[0])
            centroid_x = np.mean(coords[1])
            centroids.append((centroid_x, centroid_y))  # Store as (x, y) for OpenCV

        # Sort points by y-coordinate (second element)
        centroids_sorted_by_y = sorted(centroids, key=lambda p: p[1])

        # Get top two and bottom two points
        top_two = centroids_sorted_by_y[:2]
        bottom_two = centroids_sorted_by_y[2:]

        # Sort each pair by x-coordinate (first element)
        top_left, top_right = sorted(top_two, key=lambda p: p[0])
        bottom_left, bottom_right = sorted(bottom_two, key=lambda p: p[0])

        # Create destination points array in the correct order
        dst_points = np.array([
            top_left,       # Top-left
            top_right,      # Top-right
            bottom_left,    # Bottom-left
            bottom_right    # Bottom-right
        ], dtype=np.float32)

        # Create source points from the foreground (original image)
        h, w = fg_np.shape[:2]  # Size of foreground
        src_points_variants = {
            "0": np.array([
                [0, 0],        # Top-left
                [w-1, 0],      # Top-right
                [0, h-1],      # Bottom-left
                [w-1, h-1]     # Bottom-right
            ], dtype=np.float32),

            "+90": np.array([
                [0, h-1],      # Bottom-left
                [0, 0],        # Top-left
                [w-1, h-1],    # Bottom-right
                [w-1, 0]       # Top-right
            ], dtype=np.float32),

            "+180": np.array([
                [w-1, h-1],    # Bottom-right
                [0, h-1],      # Bottom-left
                [w-1, 0],      # Top-right
                [0, 0]         # Top-left
            ], dtype=np.float32),

            "+270": np.array([
                [w-1, 0],      # Top-right
                [w-1, h-1],    # Bottom-right
                [0, 0],        # Top-left
                [0, h-1]       # Bottom-left
            ], dtype=np.float32)
        }

        # Use the selected rotation
        src_points = src_points_variants[rotate]

        # Compute perspective transformation
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply the transformation to foreground
        warped = cv2.warpPerspective(fg_np, M, (bg_np.shape[1], bg_np.shape[0]),
                                    flags=cv2.INTER_LANCZOS4)

        # Get the transformed corners (for creating the antialiased mask)
        corners = []
        for point in dst_points:
            corners.append((int(point[0]), int(point[1])))

        # Сортируем точки по часовой стрелке вокруг центра
        center_x = sum(p[0] for p in corners) / 4
        center_y = sum(p[1] for p in corners) / 4
        corners = sorted(corners, key=lambda p: math.atan2(p[1] - center_y, p[0] - center_x))

        # Calculate center of the quadrilateral
        center_x = sum(p[0] for p in corners) / 4
        center_y = sum(p[1] for p in corners) / 4
        center = (center_x, center_y)

        # If inset is > 0, move points inward along vectors from corners to center
        if inset > 0:
            adjusted_corners = []
            for i, corner in enumerate(corners):
                # Calculate vector from corner to center
                vec_x = center[0] - corner[0]
                vec_y = center[1] - corner[1]

                # Calculate length of this vector
                length = np.sqrt(vec_x**2 + vec_y**2)

                # Normalize vector
                if length > 0:
                    vec_x /= length
                    vec_y /= length

                # Move corner inward by inset fraction of the distance
                move_dist = length * inset
                new_x = corner[0] + vec_x * move_dist
                new_y = corner[1] + vec_y * move_dist

                adjusted_corners.append((int(new_x), int(new_y)))

            corners = adjusted_corners

        # Create an empty mask
        mask_antialiased = np.zeros((bg_np.shape[0], bg_np.shape[1]), dtype=np.uint8)

        # Draw filled polygon with antialiasing
        corners_np = np.array([corners], dtype=np.int32)
        cv2.fillPoly(mask_antialiased, corners_np, color=255, lineType=cv2.LINE_AA)

        # Normalize mask to [0,1] for blending
        mask_float = mask_antialiased.astype(np.float32) / 255.0

        # Expand dimensions for broadcasting with color channels
        mask_float = np.expand_dims(mask_float, axis=2)

        # Blend warped image with background using antialiased mask
        output = warped * mask_float + bg_np * (1.0 - mask_float)

        # Convert back to tensor
        output_tensor = torch.from_numpy(output).unsqueeze(0).float()

        return (output_tensor,)

# Node registration
FCPMASK_CLASS_MAPPINGS = {
    "FourCornerPinMaskX": FourCornerPinMaskX
}

FCPMASK_DISPLAY_NAME_MAPPINGS = {
    "FourCornerPinMaskX": "Four Corner Pin by Mask X"
}