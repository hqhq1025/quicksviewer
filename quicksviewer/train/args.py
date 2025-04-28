from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import transformers

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    freeze_vision_tower: bool = field(default=True)
    tune_mm_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    # image_processor: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_patch_start_end: bool = field(default=False)
    mm_use_video_start_end: bool = field(default=False)
    # mm_use_frame_start_end: bool = field(default=False)
    # mm_use_im_patch_token: bool = field(default=True)
    mm_use_thumbnail: bool = field(default=True)
    mm_use_thumbnail_start_end: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default='spatial_pool')
    mm_spatial_pool_mode: Optional[str] = field(default="average")
    mm_spatial_pool_stride: Optional[int] = field(default=4)
    mm_spatial_pool_out_channels: Optional[int] = field(default=1024)
    mm_patch_merge_type : Optional[str] = field(default="spatial_unpad")
    mm_cubing : Optional[str] = field(default=None)
    cubing_vit_forward_n_layers : Optional[int] = field(default=-1)
    bert_type: Optional[str] = field(default="qformer_pretrain")
    num_query: Optional[int] = field(default=32)
    pretrain_qformer: Optional[str] = field(default=None)
    compress_type: Optional[str] = field(default=None)
    model_base: Optional[str] = field(default=None) # Qiji: for evaluation
    rebuild_vision: Optional[bool] = field(default=True) # Qiji: stage1


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    video_token: Optional[int] = field(default=2)
    # image_aspect_ratio: str = 'square'
    image_aspect_ratio: str = 'anyres'
    # image_grid_pinpoints: Optional[str] = field(default=None)
    image_grid_pinpoints : Optional[str] = field(default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    input_prompt: Optional[str] = field(default=None)
    refine_prompt: Optional[bool] = field(default=False)
    packed_seq_len: Optional[int] = field(default=2048*16)
    video_num_frames: Optional[int] = field(default=32)
    video_fps: Optional[int] = field(default=1)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    num_train_epochs: Optional[int] = field(default=3)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    group_by_modality_length: bool = field(default=False)
    lr_multi: Optional[str] = field(default=None)
    seq_parallel_size: int = field(
        default=-1,
        metadata={"help": "The degree of sequence parallelism (SP). SP is disabled by default (value: -1). "},
    )
    seq_parallel_ring_size: int = field(
        default=-1,
        metadata={
            "help": "The communication process group size using optimized Ring Attention approach in SP, where `seq_parallel_size` = `seq_parallel_ring_size` x `seq_parallel_ulysses_size` (determined by other two terms). Ring Attention approach is disabled by default in SP. This setting is adjustable only when `seq_parallel_size` > 1."
        },
    )
    seq_parallel_ring_type: str = field(
        default="ring_varlen",
        metadata={
            "help": "Ring Attention implementation. Support ['ring_varlen', 'zigzag_ring_varlen'] in 2D attention. Only works when `seq_parallel_ring_size` > 1."
        },
    )
    # Set learning rate for separate modules individually; default is 'lr' if not specify.
    mm_projector_lr: Optional[float] = field(default=None)
    mm_vision_tower_lr: Optional[float] = field(default=None)
    mm_resampler_lr: Optional[float] = field(default=None)
    mm_cubing_lr: Optional[float] = field(default=None)
