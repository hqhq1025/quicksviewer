#!/bin/bash
script_path=$(realpath $0)
script_dir=$(dirname $(dirname $(dirname $script_path)))

# # ----- Setup environments -----
export PYTHONPATH=$script_dir

# # ----- Setup wandb -----
# export WANDB_API_KEY=""
# export WANDB_PROJECT=quicksviewer_stage2


# ----- Setup path -----
data_dir=/data
models_dir=$data_dir/models


datasets="$data_dir/img_train_datasets/obelics_gt2.jsonl#18000_40000;$data_dir/img_train_datasets/LLaVAReCap558K.jsonl#0_50000;$data_dir/img_train_datasets/LLaVAOneVision.jsonl;$data_dir/vid_train_datasets/FineVideo/FineVideo_caption_storyline.jsonl#0_4000;$data_dir/vid_train_datasets/tsv_data/ANetCaptions/chat.jsonl#0_1000;$data_dir/vid_train_datasets/FineVideo/FineVideo_qa_climax.jsonl;$data_dir/vid_train_datasets/tsv_data/ShareGPT4Video/chat.jsonl"


train_args=" \
    --deepspeed ./scripts/zero1.json \
    --model_name_or_path $data_dir/checkpoints/quicksviewer-s1/checkpoint-14033/ \
    --version qwen2 \
    --data_path $datasets \
    --video_folder $data_dir/vid_train_datasets/tsv_data \
    --vision_tower $models_dir/siglip-so400m-patch14-384 \
    --rebuild_vision False \
    --mm_projector_type identity \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --mm_use_patch_start_end True \
    --mm_use_video_start_end True \
    --mm_use_thumbnail True \
    --mm_use_thumbnail_start_end True \
    --mm_resampler_type qformer \
    --mm_patch_merge_type flat \
    --mm_cubing momentum \
    --cubing_vit_forward_n_layers -1 \
    --image_aspect_ratio anyres \
    --video_num_frames 420 \
    --video_fps 1 \
    --compress_type "mean" \
    --bf16 True \
    --tf32 True \
    --output_dir .checkpoints/quicksviewer-s2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.02 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --freeze_backbone True \
    --freeze_mm_adapter False \
    --tune_mm_adapter True \
    --freeze_vision_tower False \
    --report_to wandb
"

    # --version llama_3_chat \

    # --seq_parallel_size 2
    # --seq_parallel_ring_size 2 \
    # --seq_parallel_ring_type ring_varlen

    # --optim adamw_torch \
    # --lr_scheduler_type "cosine" \
    # --vision_tower $modelsdir/clip-vit-large-patch14-336 \
    # --report_to none \
    # --mm_cubing identity \
    # --mm_projector_type mlp2x_gelu \
    # --mm_resampler_type spatial_pool \
    # --mm_spatial_pool_mode average \
    # --mm_spatial_pool_stride 4 \
    # --max_steps 2 \
    # --tf32 True \

train_script=train.py # no sp
# train_script=train_hybrid.py

# use deepspeed
# 单机多卡可以使用这种方式
#CMD="deepspeed --master_port 61001 quicksviewer/train/train_hybrid.py ${train_args}"

# use torchrun
# 多机多卡使用这种方式
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# 支持单机多卡和多机多卡 DP 时会没有以下环境
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-12345}

# CMD="torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --node_rank=${RANK} --master_addr=${MASTER_ENDPOINT} --master_port=${MASTER_PORT} quicksviewer/train/${train_script} ${train_args}"
CMD="torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} quicksviewer/train/${train_script} ${train_args}"


echo "-------final CMD is------"
echo "${CMD}"
echo "-------final CMD end------"

$CMD
