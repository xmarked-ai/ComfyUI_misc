import torch
from torchvision import transforms
import PIL.Image
import numpy as np
import math

class RelightX:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "normals": ("IMAGE",),
                "depth": ("IMAGE",),
                "light_x": ("FLOAT", {"default": 0.0, "min": -10, "max": 10, "step": 0.01}),
                "light_y": ("FLOAT", {"default": 0.0, "min": -10, "max": 10, "step": 0.01}),
                "light_z": ("FLOAT", {"default": -1.0, "min": -10, "max": 10, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100, "step": 0.01}),
                "specular_brightness": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 100, "step": 0.01}),
                "specular_size": ("FLOAT", {"default": 50, "min": 0.0, "max": 500, "step": 0.01}),
                "light_type": ("BOOLEAN", {"default": False, "label_on": "point light", "label_off": "directional light", "forceInput": False}),
                "invert_x_normals": ("BOOLEAN", {"default": False, "label_on": "X inverted", "label_off": "marigold compat", "forceInput": False}),
                "light_wrap": ("FLOAT", {"default": 0.0, "min": 0, "max": 10, "step": 0.01}),
                "light_wrap_type": ("BOOLEAN", {"default": False, "label_on": "gamma", "label_off": "shift", "forceInput": False}),
                "depth_of_depth": ("FLOAT", {"default": 1, "min": 0.01, "max": 10, "step": 0.01}),
                "env_gain": ("FLOAT", {"default": 1, "min": 0.01, "max": 5, "step": 0.01}),
                "env_gamma": ("FLOAT", {"default": 1, "min": 0.01, "max": 5, "step": 0.01}),
                "env_shift_x": ("INT", {"default": 0, "min": -200, "max": 200, "step": 1}),
                "env_shift_y": ("INT", {"default": 0, "min": -200, "max": 200, "step": 1}),
            },
            "optional": {
                "environment_map": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("IMAGE", "light", "points", "unit_sphere")
    FUNCTION = "relightx"

    CATEGORY = "image/filters"

    def relightx(self, image, normals, depth, light_x, light_y, light_z, brightness=1, specular_brightness=0.75, specular_size=50, light_type=False, invert_x_normals=False, light_wrap=0, light_wrap_type=False, depth_of_depth=1, environment_map=None, env_gamma=1, env_gain=1, env_shift_x=0, env_shift_y=0):
        if image.shape[0] != normals.shape[0]:
            raise Exception("Batch size for image and normals must match")

        h, w = image.shape[1:3]
        cmin = min(w,h)
        cmax = max(w,h)

        norm = normals.detach().clone() * 2 - 1
        if invert_x_normals:
            norm[:, :, :, 0] = -norm[:, :, :, 0]

        norm[:, :, :, 2] = -norm[:, :, :, 2]
        points = depth.detach().clone()

        camera = torch.tensor([0, 0, -1])
        camera = camera.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, 3)

        x = torch.linspace(-w/2, w/2, w)
        y = torch.linspace(h/2, -h/2, h)
        X, Y = torch.meshgrid(x, y)

        X = X.T.unsqueeze(0)
        Y = Y.T.unsqueeze(0)

        points[:, :, :, 0] = X / cmin * 2
        points[:, :, :, 1] = Y / cmin * 2
        points[:, :, :, 2] = (1 - points[:, :, :, 2]) * depth_of_depth

        light = torch.tensor([light_x, light_y, light_z])
        light = light.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, 3)

        light_vectors = light

        if light_type:
            light_vectors = light - points

        light_normalized = light_vectors / light_vectors.norm(dim=3, keepdim=True)
        normals_normalized = norm / norm.norm(dim=3, keepdim=True)

        diffuse_component = torch.sum(light_normalized * normals_normalized, dim=-1)

        if light_wrap > 0:
            if light_wrap_type:
                diffuse_component = torch.pow((1+diffuse_component)/2, 1/light_wrap);
            else:
                diffuse_component = 1 / (1 + light_wrap/10) * (diffuse_component+light_wrap/10);

        diffuse_component = torch.clamp(diffuse_component, min=0.0, max=1.0) * brightness
        diffuse_component = diffuse_component.unsqueeze(-1)  # (batch_size, width, height, 1)
        diffuse = diffuse_component.repeat(1, 1, 1, 3)  # expant to three channels (R, G, B)

        specular_component = (light_normalized * normals_normalized).sum(dim=-1)
        specular_component = torch.clamp(specular_component, min=0.0)
        specular_component = torch.pow(specular_component, specular_size) * specular_brightness
        specular_component = specular_component.unsqueeze(-1)
        specular = specular_component.repeat(1, 1, 1, 3)

        diffuse_with_specular = diffuse + specular

        unit_sphere = torch.tensor([0, 0, 0])
        unit_sphere = unit_sphere.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        size = 200
        relit = image.detach().clone()

        if environment_map != None:
            env_map_permuted = environment_map.detach().clone()
            env_map_squeezed = env_map_permuted.squeeze(0).permute(2, 0, 1)
            resize_transform = transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC)
            resized_env_map = resize_transform(env_map_squeezed)
            env_map = torch.clamp(resized_env_map.permute(1, 2, 0), min=0, max=1)

            env_map *= env_gain
            env_map = torch.pow(env_map, 1 / env_gamma)

            if env_shift_x:
                env_map = torch.roll(env_map, shifts=env_shift_x, dims=1)
            if env_shift_y:
                env_map = torch.roll(env_map, shifts=-env_shift_y, dims=0)

            X = torch.linspace( -1, 1, size)
            Y = torch.linspace( 1, -1, size)
            x, y = torch.meshgrid(X, Y, indexing="xy")
            z = (x **2 + y **2)
            z = torch.where(z > 1, torch.zeros_like(z), torch.sqrt(1-z))

            sh1 = torch.full((size, size), 0.282095)
            sh2 = torch.full((size, size), 0.488603)
            sh3 = torch.full((size, size), 0.488603)
            sh4 = torch.full((size, size), 0.488603)
            sh5 = torch.full((size, size), 1.092548)
            sh6 = torch.full((size, size), 1.092548)
            sh7 = torch.full((size, size), 0.315392)
            sh8 = torch.full((size, size), 1.092548)
            sh9 = torch.full((size, size), 0.546274)

            sh2 *= y
            sh3 *= z
            sh4 *= x
            sh5 *= (y * x)
            sh6 *= (y * z)
            sh7 *= (3 * z**2 - 1)
            sh8 *= (z * x)
            sh9 *= (x**2 - y**2)

            k = (sh3 > 0).float()
            sh1 *= k
            sh2 *= k
            sh3 *= k
            sh4 *= k
            sh5 *= k
            sh6 *= k
            sh7 *= k
            sh8 *= k
            sh9 *= k

            factor = 4 * math.pi / size**2

            L00  = ((env_map[:, :, 0] * sh1).sum() * factor, (env_map[:, :, 1] * sh1).sum() * factor, (env_map[:, :, 2] * sh1).sum() * factor)
            L1_1 = ((env_map[:, :, 0] * sh2).sum() * factor, (env_map[:, :, 1] * sh2).sum() * factor, (env_map[:, :, 2] * sh2).sum() * factor)
            L10  = ((env_map[:, :, 0] * sh3).sum() * factor, (env_map[:, :, 1] * sh3).sum() * factor, (env_map[:, :, 2] * sh3).sum() * factor)
            L11  = ((env_map[:, :, 0] * sh4).sum() * factor, (env_map[:, :, 1] * sh4).sum() * factor, (env_map[:, :, 2] * sh4).sum() * factor)
            L2_2 = ((env_map[:, :, 0] * sh5).sum() * factor, (env_map[:, :, 1] * sh5).sum() * factor, (env_map[:, :, 2] * sh5).sum() * factor)
            L2_1 = ((env_map[:, :, 0] * sh6).sum() * factor, (env_map[:, :, 1] * sh6).sum() * factor, (env_map[:, :, 2] * sh6).sum() * factor)
            L20  = ((env_map[:, :, 0] * sh7).sum() * factor, (env_map[:, :, 1] * sh7).sum() * factor, (env_map[:, :, 2] * sh7).sum() * factor)
            L21  = ((env_map[:, :, 0] * sh8).sum() * factor, (env_map[:, :, 1] * sh8).sum() * factor, (env_map[:, :, 2] * sh8).sum() * factor)
            L22  = ((env_map[:, :, 0] * sh9).sum() * factor, (env_map[:, :, 1] * sh9).sum() * factor, (env_map[:, :, 2] * sh9).sum() * factor)

            c1 = 0.429043
            c2 = 0.511664
            c3 = 0.743125
            c4 = 0.886227
            c5 = 0.247708

            r = L00[0].item()*sh1 + L1_1[0].item()*sh2 + L10[0].item()*sh3 + L11[0].item()*sh4 + L2_2[0].item()*sh5 + L2_1[0].item()*sh6 + L20[0].item()*sh7 + L21[0].item()*sh8 + L22[0].item()*sh9
            g = L00[1].item()*sh1 + L1_1[1].item()*sh2 + L10[1].item()*sh3 + L11[1].item()*sh4 + L2_2[1].item()*sh5 + L2_1[1].item()*sh6 + L20[1].item()*sh7 + L21[1].item()*sh8 + L22[1].item()*sh9
            b = L00[2].item()*sh1 + L1_1[2].item()*sh2 + L10[2].item()*sh3 + L11[2].item()*sh4 + L2_2[2].item()*sh5 + L2_1[2].item()*sh6 + L20[2].item()*sh7 + L21[2].item()*sh8 + L22[2].item()*sh9

            img_tensor = torch.stack((r, g, b), dim=0).permute(1, 2, 0)
            unit_sphere = torch.clamp(img_tensor, 0, 1).unsqueeze(0)

            norm_map = (normals.detach().clone() * 2 - 1).squeeze(0)
            xx = norm_map[:, :, 0].squeeze(0)
            yy = norm_map[:, :, 1].squeeze(0)
            zz = norm_map[:, :, 2].squeeze(0)
            norm_map = norm_map.unsqueeze(0)

            r = c1*L22[0].item()*(xx-yy) + c3*L20[0].item()*zz*zz + c4*L00[0].item() - c5*L20[0].item() + 2*c1*(L2_2[0].item()*xx*yy+L21[0].item()*xx*zz+L2_1[0].item()*yy*zz) + 2*c2*(L11[0].item()*xx+L1_1[0].item()*yy+L10[0].item()*zz)
            g = c1*L22[1].item()*(xx-yy) + c3*L20[1].item()*zz*zz + c4*L00[1].item() - c5*L20[1].item() + 2*c1*(L2_2[1].item()*xx*yy+L21[1].item()*xx*zz+L2_1[1].item()*yy*zz) + 2*c2*(L11[1].item()*xx+L1_1[1].item()*yy+L10[1].item()*zz)
            b = c1*L22[2].item()*(xx-yy) + c3*L20[2].item()*zz*zz + c4*L00[2].item() - c5*L20[2].item() + 2*c1*(L2_2[2].item()*xx*yy+L21[2].item()*xx*zz+L2_1[2].item()*yy*zz) + 2*c2*(L11[2].item()*xx+L1_1[2].item()*yy+L10[2].item()*zz)

            img_tensor = torch.stack((r, g, b), dim=0).permute(1, 2, 0).unsqueeze(0)
            relit[:,:,:,:3] = torch.clip(relit[:,:,:,:3] * img_tensor, 0, 1)

        if brightness > 0 or specular_brightness > 0:
            relit[:,:,:,:3] = torch.clip(relit[:,:,:,:3] * diffuse_with_specular, 0, 1)

        return (relit, diffuse_with_specular, points, unit_sphere )

RELIGHT_CLASS_MAPPINGS = {
    "RelightX": RelightX,
}

RELIGHT_NAME_MAPPINGS = {
    "RelightX": "RelightX",
}
