import os
import subprocess
from tqdm import tqdm


def extract_first_frame(video_path, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    # Using ffmpeg to extract the first frame
    command = [
        'ffmpeg', '-i', video_path, '-vf', 'select=eq(n\,0)', '-q:v', '3', 
        output_path, '-y'
    ]
    # It is recommended that you check stdout and err before you direct them to NULL like I do
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def process_videos_walkdir(base_dir, output_base):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, base_dir)
                output_dir = os.path.join(output_base, relative_path)
                output_file = os.path.join(output_dir, file.replace('.mp4', '.jpg'))
                extract_first_frame(video_path, output_file)
                

def walkdir_way():
    # Base paths for the original videos
    data_dir = '/data1/yfliu/voxceleb2/'
    dev_path = data_dir+'dev'
    test_path = data_dir+'test'

    # Output base paths for the frames
    frames_dev_path = data_dir+'frames/dev'
    frames_test_path = data_dir+'frames/test'

    # Process videos and extract frames
    process_videos_walkdir(dev_path, frames_dev_path)
    process_videos_walkdir(test_path, frames_test_path)


def save_tsv(tsvin_dir, outputbase_dir, tsvout_dir):
    os.makedirs(outputbase_dir, exist_ok=True)
    writer = open(tsvout_dir, 'w')
    writer.write('/\n')
    with open(tsvin_dir, 'r') as f:
        lines = f.readlines()[1:]
        pbar = tqdm(lines)
        for line in pbar:
            splits = line.split('\t')
            rel_path = splits[0]
            video_path = splits[1]
            output_file = os.path.join(outputbase_dir, rel_path)+'.jpg'
            writer.write(f'{rel_path}\t{output_file}\t{video_path}\n')
    writer.close()


def tsv_way(split):
    print(f'doing {split}')
    tsvindir = f"/data1/yfliu/lrs3/433h_data/{split}.tsv"
    tsvoutdir = f"/data1/yfliu/lrs3/433h_data/frame_{split}.tsv"
    outputbase_dir = "/data1/yfliu/lrs3/frames"
    save_tsv(tsvindir, outputbase_dir, tsvoutdir)
    print(f"{split} finished")


if __name__ == '__main__':
    for split in ['trainval', 'test', 'short-pretrain']:
        process_videos_walkdir(f'/data1/yfliu/lrs3/{split}', f'/data1/yfliu/lrs3/frames/{split}')
    tsv_way("train")
    tsv_way("valid")
    tsv_way("test")
