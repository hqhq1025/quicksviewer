import os
from quicksviewer.model.multimodal_encoder.clip_encoder import CLIPVisionTower
# from llavacube.model.multimodal_encoder.siglip_encoder import SiglipVisionTower
from quicksviewer.model.multimodal_encoder.siglip_encoder_custom import SiglipVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "siglip" in vision_tower:
        return SiglipVisionTower(vision_tower, vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
