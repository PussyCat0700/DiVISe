from ravdess_paths import AUDIO_RESAMPLED_HUBERT_TSV, KMEANS_RESAMPLED_HUBERT_TSV, FRAMES_SOURCE, TEST_HUBERT_TSV, TEST_TSV, TRAIN_HUBERT_TSV, TRAIN_TSV, VALID_HUBERT_TSV, VALID_TSV
import os
import random


class RAVDESSDataHandler:
    def __init__(self, audio_resampled_tsv, km_resampled_path, frames_source):
        self.audio_resampled_tsv = audio_resampled_tsv
        self.kmeans_tsv = km_resampled_path
        self.frames_source = frames_source
        self.train_file = TRAIN_TSV
        self.valid_file = VALID_TSV
        self.test_file = TEST_TSV
        self.train_hubert_file = TRAIN_HUBERT_TSV
        self.valid_hubert_file = VALID_HUBERT_TSV
        self.test_hubert_file = TEST_HUBERT_TSV
        
        # Open files for writing
        self.train_fw = open(self.train_file, 'w')
        self.valid_fw = open(self.valid_file, 'w')
        self.test_fw = open(self.test_file, 'w')
        # HuBERT tsv files
        self.train_hubert_fw = open(self.train_hubert_file, 'w')
        self.valid_hubert_fw = open(self.valid_hubert_file, 'w')
        self.test_hubert_fw = open(self.test_hubert_file, 'w')

    def remove_prefix(self, path):
        index = path.find('Actor_')
        if index != -1:
            return path[index:]
        return path

    def extract_info(self, filename):
        parts = filename.split('-')
        emotion = parts[2]            # Emotion
        intensity = parts[3]          # Emotional intensity
        statement = parts[4]          # Statement
        actor = parts[6]              # Actor number
        gender = '02' if int(actor) % 2 == 0 else '01'
        return emotion, intensity, statement, actor, gender

    def process(self):
        # Step 1: Collect data per actor
        data_by_actor = {}
        with open(self.audio_resampled_tsv, 'r') as fr:
            for i, line in enumerate(fr.readlines()):
                if i == 0:
                    self.fr_km = open(self.kmeans_tsv, 'r')
                    continue
                
                wav_path, wav_len = line.strip().split('\t')
                raw_rel_wav_path = self.remove_prefix(wav_path)
                actor_prefix, wav_filename = raw_rel_wav_path.split(os.sep)
                frame_raw_path = '01' + wav_filename.replace('.wav', '.jpg')[2:]
                frame_path = os.path.join(self.frames_source, f"Video_Speech_{actor_prefix}", actor_prefix, frame_raw_path)
                emotion, intensity, statement, actor, gender = self.extract_info(wav_filename.replace('.wav', ''))

                # Collect data per actor
                if actor not in data_by_actor:
                    data_by_actor[actor] = []
                data_by_actor[actor].append((frame_path, emotion, intensity, statement, actor, gender, wav_path, wav_len, self.fr_km.readline()))

        # Step 2: Split data for each actor into train/valid/test
        for actor, data in data_by_actor.items():
            total_count = len(data)
            train_count = int(0.7 * total_count)
            valid_count = int(0.2 * total_count)

            random.shuffle(data)  # Shuffle data to ensure random splits
            train_data = data[:train_count]
            valid_data = data[train_count:train_count + valid_count]
            test_data = data[train_count + valid_count:]

            # Step 3: Write each part to the respective files
            self.write_data(train_data, self.train_fw, self.train_hubert_fw)
            self.write_data(valid_data, self.valid_fw, self.valid_hubert_fw)
            self.write_data(test_data, self.test_fw, self.test_hubert_fw)

    def write_data(self, data, fw, fw_km):
        for (frame_path, emotion, intensity, statement, actor, gender, wav_path, wav_len, km_line) in data:
            fw.write(f'{frame_path}\t{emotion}\t{intensity}\t{statement}\t{actor}\t{gender}\t{wav_path}\t{wav_len}\n')
            fw_km.write(km_line)

    def close(self):
        self.fr_km.close()
        self.train_fw.close()
        self.valid_fw.close()
        self.test_fw.close()
        self.train_hubert_fw.close()
        self.valid_hubert_fw.close()
        self.test_hubert_fw.close()


handler = RAVDESSDataHandler(AUDIO_RESAMPLED_HUBERT_TSV, KMEANS_RESAMPLED_HUBERT_TSV, FRAMES_SOURCE)
handler.process()
handler.close()
