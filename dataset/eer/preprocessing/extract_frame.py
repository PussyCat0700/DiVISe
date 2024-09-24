import os
import subprocess
from tqdm import tqdm


# Function to get the total number of frames using ffprobe
def get_total_frames(video_path):
    command = [
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', 
        '-show_entries', 'stream=nb_read_frames', '-of', 'default=nokey=1:noprint_wrappers=1', video_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return int(result.stdout.decode().strip())

# Function to extract the middle frame using ffmpeg
def extract_middle_frame(video_path, output_path):
    total_frames = get_total_frames(video_path)
    middle_frame = total_frames // 2  # Get the middle frame index
    # Construct ffmpeg command to extract the middle frame
    command = [
        'ffmpeg', '-i', video_path, '-vf', f'select=eq(n\\,{middle_frame})', '-frames:v', '1', '-q:v', '3',
        output_path, '-y'
    ]
    # It is recommended that you check stdout and err before you direct them to NULL like I do now
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def extract_first_frame(video_path, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    # Using ffmpeg to extract the first frame
    command = [
        'ffmpeg', '-i', video_path, '-vf', 'select=eq(n\,0)', '-q:v', '3', 
        output_path, '-y'
    ]
    # It is recommended that you check stdout and err before you direct them to NULL like I do now
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def process_videos_walkdir(base_dir, output_base, func=extract_first_frame):
    os.makedirs(output_base, exist_ok=True)
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, base_dir)
                output_dir = os.path.join(output_base, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, file.replace('.mp4', '.jpg'))
                func(video_path, output_file)
                

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


def tsv_way(inbasedir, outputbase_dir, split):
    print(f'doing {split}')
    tsvindir = f"{inbasedir}/{split}.tsv"
    tsvoutdir = f"{inbasedir}/frame_{split}.tsv"
    save_tsv(tsvindir, outputbase_dir, tsvoutdir)
    print(f"{split} finished")


if __name__ == '__main__':
    def make_lrs3_tsv():
        for split in ['trainval', 'test', 'short-pretrain']:
            process_videos_walkdir(f'/data1/yfliu/lrs3/{split}', f'/data1/yfliu/lrs3/frames/{split}', func=extract_middle_frame)
        tsv_way("/data1/yfliu/lrs3/433h_data", "/data1/yfliu/lrs3/frames", "train")
        tsv_way("/data1/yfliu/lrs3/433h_data", "/data1/yfliu/lrs3/frames", "valid")
        tsv_way("/data1/yfliu/lrs3/433h_data", "/data1/yfliu/lrs3/frames", "test")
        
    def make_lrs2_tsv():
        for split in ['main', 'short-pretrain']:
            process_videos_walkdir(f'/data1/yfliu/lrs2ondisk/mvlrs_v1/{split}', f'/data1/yfliu/lrs2ondisk/mvlrs_v1/frames/{split}')
        tsv_way("/data1/yfliu/lrs2ondisk/mvlrs_v1/224h_data", "/data1/yfliu/lrs2ondisk/mvlrs_v1/frames", "train")
        tsv_way("/data1/yfliu/lrs2ondisk/mvlrs_v1/224h_data", "/data1/yfliu/lrs2ondisk/mvlrs_v1/frames", "valid")
        tsv_way("/data1/yfliu/lrs2ondisk/mvlrs_v1/224h_data", "/data1/yfliu/lrs2ondisk/mvlrs_v1/frames", "test")
    
    def make_vox2_tsv():
        tsv_indir = '/data1/yfliu/voxceleb2/all_data/test.tsv'
        outputbase_dir = "/data1/yfliu/voxceleb2/frames"
        output_tsv = '/data1/yfliu/voxceleb2/all_data/frame_test.tsv'
        save_tsv(tsv_indir, outputbase_dir, output_tsv)
    
    def make_ravdess_tsv():
        outputbase_dir = "/data1/yfliu/ravdess/frames"
        process_videos_walkdir('/data1/yfliu/ravdess', outputbase_dir, func=extract_middle_frame)

    make_lrs3_tsv()