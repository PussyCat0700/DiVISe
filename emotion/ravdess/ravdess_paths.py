import os
# Define the source and target directories
AUDIO_SOURCE = "/data1/yfliu/ravdess/Audio_Speech_Actors_01-24"
AUDIO_RESAMPLED = "/data1/yfliu/ravdess/Resampled/Audio_Speech_Actors_01-24"
AUDIO_RESAMPLED_HUBERT_TSV = os.path.join(AUDIO_RESAMPLED, "hubert.tsv")
KMEANS_RESAMPLED_HUBERT_TSV = os.path.join(AUDIO_RESAMPLED, "hubert.km")
FRAMES_SOURCE = "/data1/yfliu/ravdess/frames"
FINAL_TSV_PATH = "/data1/yfliu/ravdess"
# Splits
TRAIN_HUBERT_TSV = os.path.join(FINAL_TSV_PATH, "train_hubert.tsv")
VALID_HUBERT_TSV = os.path.join(FINAL_TSV_PATH, "valid_hubert.tsv")
TEST_HUBERT_TSV = os.path.join(FINAL_TSV_PATH, "test_hubert.tsv")
TRAIN_TSV = os.path.join(FINAL_TSV_PATH, "train.tsv")
VALID_TSV = os.path.join(FINAL_TSV_PATH, "valid.tsv")
TEST_TSV = os.path.join(FINAL_TSV_PATH, "test.tsv")
