#!/bin/bash
script_path=$(realpath $0)
script_dir=$(dirname $(dirname $(dirname $script_path)))

export CUDA_LAUNCH_BLOCKING=1 # This for prevent CUDA crash, which causes slowing down!!!
export PYTHONPATH=$script_dir


data_dir=/data
models_dir=$data_dir/models


# ------- Settings -------------
TASK=${1-"videomme"}
MODELPATH=${2:-"checkpoints/quicksviewer-s3/checkpoint-15408"}
NFRAMES=${3:-420}
FPS=${4:-1}
SAVEDIR=$data_dir/results/quicksviewer
if [ $# -ge 2 ]; then
    RANGES=("${@: 5:4}") # specify ranges manually
fi


if [ "$TASK" == "mvbench" ]; then
    # MVBench
    script_py=eval_mvbench.py
    data_path=$data_dir/MVBench/MVBench/json/
    video_dir=$data_dir/MVBench/MVBench/video/
    ranges=(${RANGES[@]:-"0_1000" "1000_2000" "2000_3000" "3000_-1"})
elif [ "$TASK" == "mlvu" ]; then
    # MLVU
    script_py=eval_mlvu.py
    data_path=$data_dir/MVLU/MLVU/
    video_dir="none"
    ranges=(${RANGES[@]:-"0_543" "543_1086" "1086_1629" "1629_-1"})
elif [ "$TASK" == "videoChat_generic_qa" ] || [ "$TASK" == "videoChat_consistency_qa" ] || [ "$TASK" == "videoChat_temporal_qa" ]; then
    # Video-Chatgpt-QA
    script_py=eval_videochatgptqa.py
    data_path=$data_dir/Video-ChatGPT/
    video_dir=$data_dir/Video-ChatGPT/activity/videos/
    if [ "$TASK" == "videoChat_generic_qa" ]; then
        ranges=(${RANGES[@]:-"0_500" "500_1000" "1000_1500" "1500_-1"})
    else
        ranges=(${RANGES[@]:-"0_125" "125_250" "250_375" "375_-1"})
    fi
elif [ "$TASK" == "nextqa" ]; then
    # NExT-QA
    script_py=eval_nextqa.py
    data_path=$data_dir/NExT-QA/test-data-nextqa/test.csv
    video_dir=$data_dir/nextqa_videos/
    ranges=(${RANGES[@]:-"0_2141" "2141_4282" "4282_6423" "6423_-1"})
elif [ "$TASK" == "activitynetqa" ]; then
    script_py=eval_activitynet_qa.py
    data_path=$data_dir/activitynet-qa/dataset
    video_dir=$data_dir/ActivityNetQA/all_test
    ranges=(${RANGES[@]:-"0_2000" "2000_4000" "4000_6000" "6000_-1"})
else
    # VideoMME (default)
    script_py=eval_video_mme.py
    data_path=$data_dir/Video-MME/videomme/test-00000-of-00001.parquet
    video_dir=$data_dir/Video-MME/video_mme_video/
    ranges=(${RANGES[@]:-"0_700" "700_1400" "1400_2100" "2100_-1"})
fi
echo "Runing evaluation for sample ranges: ${ranges[@]}"

# ------- Run Evaluate -------------
# for i in {0..0}
# for i in $(seq 0 2 6)
seq=($(seq 0 2 6))
n=${#ranges[@]}
devices=("${seq[@]:0:$n}") # get first n elements
for i in ${devices[@]}
do
    # echo $script_py $data_path $video_dir ${ranges[@]}
    CUDA_VISIBLE_DEVICES="$i,$((i+1))" python $script_dir/quicksviewer/eval/$script_py --model-path $MODELPATH --data_path $data_path --video_dir $video_dir --task $TASK --samples_range ${ranges[$((i / 2))]} --video_nframes $NFRAMES --video_fps $FPS --save_dir $SAVEDIR &
done
wait
echo "Done for evaluation."


#  -------  Calculate Metrics  ------- 
python $script_dir/quicksviewer/eval/$script_py --model-path $MODELPATH --data_path $data_path --video_dir $video_dir --task $TASK --samples_range ${ranges[1]} --video_nframes $NFRAMES --video_fps $FPS --save_dir $SAVEDIR --calc_metrics
echo "Done for calculating metrics."