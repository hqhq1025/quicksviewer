#!/bin/bash
script_path=$(realpath $0)
script_dir=$(dirname $(dirname $(dirname $script_path)))
export PYTHONPATH=$script_dir

data_dir=/data
num_workers=${1:-100}
samples_range=${2:-"0_-1"}


python quicksviewer/preprocess/convert_sharegpt4video.py \
    --ann_path $data_dir/ShareGPT4Video/sharegpt4video_40k.jsonl \
    --video_dir $data_dir/sharegpt4video/videos/ \
    --output_dir $data_dir/vid_train_datasets/tsv_data/ShareGPT4Video \
    --num_workers $num_workers
echo "Done for processing ShareGPT4Video!"
