import os
import librosa
import soundfile as sf

from ravdess_paths import AUDIO_SOURCE, AUDIO_RESAMPLED, AUDIO_RESAMPLED_HUBERT_TSV

# Function to resample audio to 16kHz
def resample_audio(source_path, target_path, sr=16000):
    audio, original_sr = librosa.load(source_path, sr=None)  # Load with original sample rate
    resampled_audio = librosa.resample(audio, orig_sr=original_sr, target_sr=sr)  # Resample to 16kHz
    os.makedirs(os.path.dirname(target_path), exist_ok=True)  # Create target directory if it doesn't exist
    sf.write(target_path, resampled_audio, sr)  # Save the resampled audio
    return len(resampled_audio)

if os.path.exists(AUDIO_RESAMPLED_HUBERT_TSV):
    # remove check at your own risk of overwrite.
    print(f'{AUDIO_RESAMPLED_HUBERT_TSV} already exists!')
    exit(0)
fw = open(AUDIO_RESAMPLED_HUBERT_TSV, 'w')
fw.write('/\n')
# Loop through all files in the source directory and resample
for actor in os.listdir(AUDIO_SOURCE):
    actor_dir = os.path.join(AUDIO_SOURCE, actor)
    if os.path.isdir(actor_dir):
        for file in os.listdir(actor_dir):
            if file.endswith(".wav"):
                source_file_path = os.path.join(actor_dir, file)
                target_file_path = os.path.join(AUDIO_RESAMPLED, actor, file)
                len_resampled_audio = resample_audio(source_file_path, target_file_path)
                fw.write(f'{target_file_path}\t{len_resampled_audio}\n')
fw.close()
print("Resampling & exporting complete!")
