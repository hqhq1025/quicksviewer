import sys,os
from tqdm import tqdm
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg

from quicksviewer.utils.mm_utils import is_video

def get_seconds(str_timecode):
    splits = str_timecode.split(':')
    hours, minutes, seconds = int(splits[0]), int(splits[1]), float(splits[2])
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds


def get_video_list(video_root):
    # video_root = '/root'
    video_names = os.listdir(video_root)
    video_paths = []
    for video_name in video_names:
        if is_video(video_name):
            video_path = os.path.join(video_root, video_name)
            video_paths.append(video_path)
    return video_paths


def process_videos(video_paths, out_dir):

    for video_index, video_path in tqdm(enumerate(video_paths)):
        # video_name = os.path.basename(video_path)
        print('process {}th of {} videos'.format(video_index, len(video_paths)))
        try:
            scene_list = detect(video_path, AdaptiveDetector())
            filter_list = []
            for scene in scene_list:
                start_time, end_time = scene[0].get_timecode(), scene[1].get_timecode()
                float_start, float_end = get_seconds(start_time), get_seconds(end_time)
                if float_end - float_start >= 5:
                    filter_list.append(scene)

            if len(filter_list) > 0:
                split_video_ffmpeg(video_path, filter_list, output_dir=out_dir, arg_override="-map 0:v -map 0:a? -map 0:s?")
        except:
            print('Error processing {}'.format(video_path))
        

if __name__ == '__main__':
    video_root = 'activitynet/train/'
    output_dir = 'ShareGPTVideo/activitynet_train'
    video_paths = get_video_list(video_root)
    process_videos(video_paths, output_dir)


