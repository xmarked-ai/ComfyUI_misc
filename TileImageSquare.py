from nodes import MAX_RESOLUTION
import torch

class ImageTileSquare:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image":       ("IMAGE",),
                "tile_size":   ("INT", { "default": 768, "min": 1, "max": 2048, "step": 1, }),
                "min_overlap": ("INT", { "default": 64, "min": 1, "max": 1000, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "tile_size", "overlap_x", "overlap_y", "cols", "rows", "scr_w", "src_h", "croped_w", "croped_h")
    FUNCTION = "execute"
    CATEGORY = "xmtools/image manipulation"

    def execute(self, image, tile_size, min_overlap):
        w, h = image.shape[1:3]

        cols, r = divmod(h, tile_size-min_overlap//2)
        cols+=bool(r)

        rows, r = divmod(w, tile_size-min_overlap//2)
        rows+=bool(r)

        overlap_x = -((cols * tile_size - h) // (1-cols))
        overlap_y = -((rows * tile_size - w) // (1-rows))

        crop_h = tile_size * cols - overlap_x * (cols-1)
        crop_w = tile_size * rows - overlap_y * (rows-1)

        x1 = (h-crop_h) // 2
        y1 = (w-crop_w) // 2

        x2 = x1+crop_h
        y2 = y1+crop_w

        if x2 > h:
            x2 = h
        if y2 > w:
            y2 = w

        image = image[:, y1:y2, x1:x2, :]

        tiles = []
        for j in range(rows):
            for i in range(cols):
                x1 = i * tile_size
                y1 = j * tile_size

                if i > 0:
                    x1 -= overlap_x * i
                if j > 0:
                    y1 -= overlap_y * j

                x2 = x1 + tile_size
                y2 = y1 + tile_size

                if x2 > crop_h:
                    x2 = crop_h
                if y2 > crop_w:
                    y2 = crop_w

                tiles.append(image[:, y1:y2, x1:x2, :])
        tiles = torch.cat(tiles, dim=0)

        return (tiles, tile_size, overlap_x, overlap_y, cols, rows, h, w, crop_h, crop_w)


class ImageUntileSquare:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "overlap_x": ("INT", { "default": 1, "min": 1, "max": MAX_RESOLUTION//2, "step": 1, }),
                "overlap_y": ("INT", { "default": 1, "min": 1, "max": MAX_RESOLUTION//2, "step": 1, }),
                "cols": ("INT", { "default": 2, "min": 1, "max": 256, "step": 1, }),
                "rows": ("INT", { "default": 2, "min": 1, "max": 256, "step": 1, }),
            }
        }

    # RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "INT", "INT")
    # RETURN_NAMES = ("IMAGE", "overlap_x", "overlap_y", "cols", "rows", "out_h", "out_w")
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "xmtools/image manipulation"

    def execute(self, tiles, overlap_x, overlap_y, cols, rows):
        tile_w, tile_h = tiles.shape[1:3]
        out_h = cols * tile_h - overlap_x * (cols-1)
        out_w = rows * tile_w - overlap_y * (rows-1)

        out = torch.zeros((1, out_w, out_h, tiles.shape[3]), device=tiles.device, dtype=tiles.dtype)

        for j in range(rows):
            for i in range(cols):
                x1 = i * tile_h - overlap_x * i
                y1 = j * tile_w - overlap_y * j

                x2 = x1 + tile_h
                y2 = y1 + tile_w

                mask = torch.ones((1, tile_w, tile_h), device=tiles.device, dtype=tiles.dtype)

                # feather the overlap on top
                if j > 0:
                    mask[:, :overlap_y, :] *= torch.linspace(0, 1, overlap_y, device=tiles.device, dtype=tiles.dtype).unsqueeze(1)
                # feather the overlap on bottom
                #if i < rows - 1:
                #    mask[:, -overlap_y:, :] *= torch.linspace(1, 0, overlap_y, device=tiles.device, dtype=tiles.dtype).unsqueeze(1)
                # feather the overlap on left
                if i > 0:
                    mask[:, :, :overlap_x] *= torch.linspace(0, 1, overlap_x, device=tiles.device, dtype=tiles.dtype).unsqueeze(0)
                # feather the overlap on right
                #if j < cols - 1:
                #    mask[:, :, -overlap_x:] *= torch.linspace(1, 0, overlap_x, device=tiles.device, dtype=tiles.dtype).unsqueeze(0)
                
                mask = mask.unsqueeze(-1).repeat(1, 1, 1, tiles.shape[3])
                tile = tiles[j * cols + i] * mask
                out[:, y1:y2, x1:x2, :] = out[:, y1:y2, x1:x2, :] * (1 - mask) + tile

        return(out, )

        # return(out, overlap_x, overlap_y, cols, rows, out_h, out_w)


NODE_CLASS_MAPPINGS = {
    "ImageTileSquare": ImageTileSquare,
    "ImageUntileSquare": ImageUntileSquare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageTileSquare": "Image Tile Square",
    "ImageUntileSquare": "Image Untile Square",
}
