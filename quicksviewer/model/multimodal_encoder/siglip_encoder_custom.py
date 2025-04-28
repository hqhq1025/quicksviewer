import ipdb.stdout
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipImageProcessor
from quicksviewer.model.multimodal_encoder.modeling_siglip import SiglipVisionConfig, SiglipVisionModel



class SiglipVisionTower(nn.Module):
    def __init__(self, vision_tower_name, args, delay_load=False):
        super(SiglipVisionTower, self).__init__()
        
        # model_path = "google/siglip-so400m-patch14-384"
        self.unfreeze_mm_vision_tower = not getattr(args, 'freeze_vision_tower', True)
        # base_model_name, res, interp = model_path, 384, 576
        self.delay_load = delay_load
        res, interp = 384, 576
        self.vision_tower_name = vision_tower_name
        self._image_size = res if res is not None else 512
        self._interp_size = interp
        self.is_loaded = False
        if not self.delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()
        else:
            self._hidden_size = 1152
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        self.vision_model = "siglip"
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)

        self.vision_tower.output_tokens = True

        self._hidden_size = self.vision_tower.config.hidden_size
        self._image_size = self.vision_tower.config.image_size
        self._patch_size = self.vision_tower.config.patch_size
        self.image_processor = SiglipImageProcessor.from_pretrained(
            self.vision_tower_name
        )

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

    def interpolate(self, image_features):
        if self._interp_size is None:
            return image_features

        b, num_tokens, dim = image_features.shape

        if num_tokens != self.num_patches:
            target_h = target_w = int(self._interp_size**0.5)
            h = w = int(num_tokens**0.5)

            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2).contiguous()

            image_features = F.interpolate(
                image_features.to(torch.float32),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).to(image_features.dtype)

            # Permute the dimensions back to (b, target_h, target_w, dim)
            image_features = image_features.permute(0, 2, 3, 1).contiguous()

            # Flatten the spatial dimensions (target_h, target_w) into a single dimension
            image_features = image_features.flatten(1, 2)

        return image_features

    def _forward(self, images=None, interpolate_token=576, forward_n_layers = -1, forward_nth_embeds = None,):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_features = self.vision_tower.forward(
                # images.to(device=self.device, dtype=self.dtype),
                images.to(device=self.device, dtype=self.dtype) if images is not None else None,
                output_hidden_states=True,
                forward_n_layers = forward_n_layers,
                forward_nth_embeds = forward_nth_embeds,
            ).hidden_states[-1]
            interp_features = self.interpolate(image_features)
            return interp_features
        

    def forward(self, images=None, forward_n_layers = -1, forward_nth_embeds = None,):
        if type(images) is list:
            # image_features = [self._forward(image.unsqueeze(0)) for image in images]
            image_features = [self._forward(image.unsqueeze(0), forward_n_layers=forward_n_layers, forward_nth_embeds=forward_nth_embeds) for image in images]
        else:
            # image_features = self._forward(images)
            image_features = self._forward(images, forward_n_layers=forward_n_layers, forward_nth_embeds=forward_nth_embeds)

        return image_features


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        # Dynamically infer the dtype from the first parameter, if not explicitly specified
        if hasattr(self.vision_tower, "dtype"):
            return self.vision_tower.dtype
        else:
            params = list(self.vision_tower.parameters())
            return (
                params[0].dtype if len(params) > 0 else torch.float32
            )  # Default to torch.float32 if no parameters

    @property
    def device(self):
        # Dynamically infer the device from the first parameter, if not explicitly specified
        if hasattr(self.vision_tower, "device"):
            return self.vision_tower.device
        else:
            params = list(self.vision_tower.parameters())
            return (
                params[0].device if len(params) > 0 else torch.device("cpu")
            )  # Default to CPU if no parameters

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        try:
            return self.config.hidden_size
        except:
            return self._hidden_size

    @property
    def image_size(self):  # resolution
        # return self.config.image_size
        try:
            return self.config.image_size
        except:
            return self._image_size

    @property
    def patch_size(self):
        # return self.config.patch_size
        try:
            return self.config.patch_size
        except:
            return self._patch_size

    @property
    def num_patches_per_side(self):
        if self._interp_size is not None:
            return int(self._interp_size**0.5)
        try:
            return self.image_size // self.patch_size
        except:
            return self._num_patches_per_side

    @property
    def num_patches(self):
        if self._interp_size is not None:
            return self._interp_size
        try:
            return self.num_patches_per_side**2
        except:
            return self._num_patches