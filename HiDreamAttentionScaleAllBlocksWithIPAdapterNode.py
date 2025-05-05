import torch
import torch.nn as nn
import copy
import logging

class HiDreamAttentionScaleAllBlocksWithIPAdapterNode:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "cond_embeddings": ("CLIP_VISION_OUTPUT",),
            },
        }

        for i in range(16):
            inputs["required"][f"scale_double_text_{i}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01})
        for i in range(16):
            inputs["required"][f"scale_double_image_{i}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01})
        for i in range(32):
            inputs["required"][f"scale_single_block_{i}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01})
        return inputs

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "scale_attention_with_ipadapter"
    CATEGORY = "HiDream"

    def scale_attention_with_ipadapter(self, model, cond_embeddings=None, **kwargs):
        if not hasattr(model.model, "diffusion_model") or not hasattr(model.model.diffusion_model, "double_stream_blocks"):
            raise ValueError("Model must be a HiDream model")

        # Extract scale values for both block types
        double_text_scales = [kwargs[f"scale_double_text_{i}"] for i in range(16)]
        double_image_scales = [kwargs[f"scale_double_image_{i}"] for i in range(16)]
        single_scales = [kwargs[f"scale_single_block_{i}"] for i in range(32)]

        model_clone = copy.deepcopy(model)

        # Process conditioning embeddings if provided
        cond_emb = None
        if cond_embeddings is not None:
            if hasattr(cond_embeddings, 'last_hidden_state'):
                cond_emb = cond_embeddings.last_hidden_state
            elif hasattr(cond_embeddings, 'hidden_states'):
                cond_emb = cond_embeddings.hidden_states[-1]
            elif isinstance(cond_embeddings, torch.Tensor):
                cond_emb = cond_embeddings
            else:
                raise ValueError(f"Unsupported CLIP_VISION_OUTPUT type: {type(cond_embeddings)}")

            print(f"Original cond_emb shape: {cond_emb.shape}")

            if not isinstance(cond_emb, torch.Tensor):
                raise ValueError("cond_emb must be a tensor")

        # Define dimensions
        text_dim = 2560
        cond_dim = cond_emb.shape[-1] if cond_emb is not None else text_dim
        image_dim = model_clone.model.diffusion_model.single_stream_blocks[0].block.attn1.to_q.weight.shape[1]

        # Create shared projections for all blocks
        projection_text = None
        projection_image = None

        if cond_emb is not None:
            # Create projection from conditioning embeddings to text dimension
            projection_text = nn.Linear(cond_dim, text_dim, bias=False)
            projection_text = projection_text.to(cond_emb.device, cond_emb.dtype)
            nn.init.kaiming_uniform_(projection_text.weight, a=0.02)

            # Create projection for image dimension
            projection_image = nn.Linear(cond_dim, image_dim, bias=False)
            projection_image = projection_image.to(cond_emb.device, cond_emb.dtype)
            nn.init.kaiming_uniform_(projection_image.weight, a=0.02)

        # Patch double_stream_blocks (text + image tokens)
        for i, block in enumerate(model_clone.model.diffusion_model.double_stream_blocks):
            attn = block.block.attn1
            original_processor = attn.processor

            # Create processor with properly captured variables for double blocks
            class IPAdapterDoubleProcessor:
                def __init__(self, text_scale, image_scale, original_proc, cond_embedding=None,
                             cond_dim=cond_dim, text_dim=text_dim, image_dim=image_dim,
                             text_projection=None, image_projection=None):
                    self.text_scale = text_scale
                    self.image_scale = image_scale
                    self.original_processor = original_proc
                    self.cond_emb = cond_embedding
                    self.text_projection = text_projection
                    self.image_projection = image_projection

                def __call__(self, attn, image_tokens, image_tokens_masks=None, text_tokens=None, rope=None, *args, **kwargs):
                    if text_tokens is None:
                        return self.original_processor(attn, image_tokens, image_tokens_masks, text_tokens, rope, *args, **kwargs)

                    if self.cond_emb is None:
                        scaled_image_tokens = image_tokens * self.image_scale
                        scaled_text_tokens = text_tokens * self.text_scale
                        return self.original_processor(attn, scaled_image_tokens, image_tokens_masks,
                                                      scaled_text_tokens, rope, *args, **kwargs)

                    batch_size = image_tokens.shape[0]

                    text_seq_len = text_tokens.shape[1]
                    text_cond_emb = self.text_projection(self.cond_emb.expand(batch_size, -1, -1))
                    text_cond_emb = text_cond_emb.to(text_tokens.device, text_tokens.dtype)

                    text_cond_seq_len = text_cond_emb.shape[1]
                    if text_cond_seq_len < text_seq_len:
                        repeat_times = (text_seq_len + text_cond_seq_len - 1) // text_cond_seq_len
                        text_cond_emb = text_cond_emb.repeat(1, repeat_times, 1)[:, :text_seq_len, :]
                    elif text_cond_seq_len > text_seq_len:
                        text_cond_emb = text_cond_emb[:, :text_seq_len, :]

                    image_seq_len = image_tokens.shape[1]
                    image_cond_emb = self.image_projection(self.cond_emb.expand(batch_size, -1, -1))
                    image_cond_emb = image_cond_emb.to(image_tokens.device, image_tokens.dtype)

                    image_cond_seq_len = image_cond_emb.shape[1]
                    if image_cond_seq_len < image_seq_len:
                        repeat_times = (image_seq_len + image_cond_seq_len - 1) // image_cond_seq_len
                        image_cond_emb = image_cond_emb.repeat(1, repeat_times, 1)[:, :image_seq_len, :]
                    elif image_cond_seq_len > image_seq_len:
                        image_cond_emb = image_cond_emb[:, :image_seq_len, :]

                    combined_text_tokens = text_tokens + self.text_scale * text_cond_emb
                    combined_image_tokens = image_tokens + self.image_scale * image_cond_emb

                    return self.original_processor(attn, combined_image_tokens, image_tokens_masks,
                                                 combined_text_tokens, rope, *args, **kwargs)

            attn.processor = IPAdapterDoubleProcessor(
                double_text_scales[i],
                double_image_scales[i],
                original_processor,
                cond_emb,
                cond_dim,
                text_dim,
                image_dim,
                projection_text,
                projection_image
            )

        # Patch single_stream_blocks (image tokens only)
        for i, block in enumerate(model_clone.model.diffusion_model.single_stream_blocks):
            attn = block.block.attn1
            original_processor = attn.processor

            # Create processor with properly captured variables for single blocks
            class IPAdapterSingleProcessor:
                def __init__(self, scale_value, original_proc, cond_embedding=None,
                             cond_dim=cond_dim, image_dim=image_dim, projection=None):
                    self.scale = scale_value
                    self.original_processor = original_proc
                    self.cond_emb = cond_embedding
                    self.projection = projection

                def __call__(self, attn, image_tokens, image_tokens_masks=None, text_tokens=None, rope=None, *args, **kwargs):
                    # For single blocks, we always scale the image tokens
                    if self.cond_emb is None:
                        scaled_image_tokens = image_tokens * self.scale
                        return self.original_processor(attn, scaled_image_tokens, image_tokens_masks, text_tokens, rope, *args, **kwargs)

                    batch_size = image_tokens.shape[0]
                    image_seq_len = image_tokens.shape[1]

                    # Project and prepare conditional embeddings for image tokens
                    cond_emb_batched = self.projection(self.cond_emb.expand(batch_size, -1, -1))
                    cond_emb_batched = cond_emb_batched.to(image_tokens.device, image_tokens.dtype)

                    # Adjust sequence length to match image_tokens
                    cond_seq_len = cond_emb_batched.shape[1]
                    if cond_seq_len < image_seq_len:
                        repeat_times = (image_seq_len + cond_seq_len - 1) // cond_seq_len
                        cond_emb_batched = cond_emb_batched.repeat(1, repeat_times, 1)[:, :image_seq_len, :]
                    elif cond_seq_len > image_seq_len:
                        cond_emb_batched = cond_emb_batched[:, :image_seq_len, :]

                    # Only add scaled conditional embeddings, without scaling the base image tokens
                    combined_tokens = image_tokens + self.scale * cond_emb_batched

                    return self.original_processor(attn, combined_tokens, image_tokens_masks, text_tokens, rope, *args, **kwargs)

            attn.processor = IPAdapterSingleProcessor(
                single_scales[i],
                original_processor,
                cond_emb,
                cond_dim,
                image_dim,
                projection_image
            )

        return (model_clone,)

NODE_CLASS_MAPPINGS = {
    "HiDreamAttentionScaleAllBlocksWithIPAdapterNode": HiDreamAttentionScaleAllBlocksWithIPAdapterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiDreamAttentionScaleAllBlocksWithIPAdapterNode": "HiDream All Blocks Attention Scale + IP-Adapter"
}
