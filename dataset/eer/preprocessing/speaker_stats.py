import os
import json

def collect_speaker_data(base_path, postfix='.mp4', verbose=False):
    speaker_data = {}
    for root, dirs, files in os.walk(base_path):
        parts = root.split(os.sep)
        if len(parts) < 4:  # Not deep enough in directory structure
            continue
        speaker_id = parts[-2]
        show_id = parts[-1]
        video_files = [f for f in files if f.endswith(postfix)]
        if not video_files:
            continue
        speaker_data.setdefault(speaker_id, {})
        speaker_data[speaker_id][show_id] = {
            'path': root,
            'video_count': len(video_files),
            'videos': video_files,
        }
        if verbose:
            print(speaker_data[speaker_id][show_id])
    return speaker_data

def save_data(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    dev_path = '/data1/yfliu/voxceleb2/dev'
    test_path = '/data1/yfliu/voxceleb2/test'
    dev_data = collect_speaker_data(dev_path)
    save_data(dev_data, 'dev_speaker_data.json')
    test_data = collect_speaker_data(test_path)
    save_data(test_data, 'test_speaker_data.json')