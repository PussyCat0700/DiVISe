import json
import random
random.seed(42)


def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def generate_pairs(speaker_data, output_file):
    with open(output_file, 'w') as file:
        all_speakers = list(speaker_data.keys())
        
        for speaker_id in all_speakers:
            shows = list(speaker_data[speaker_id].keys())
            
            # Generating positive pairs
            for show_id in shows:
                videos = speaker_data[speaker_id][show_id]['videos']
                other_shows = [s for s in shows if s != show_id]
                
                for video in videos:
                    if other_shows:
                        other_show = random.choice(other_shows)
                        other_videos = speaker_data[speaker_id][other_show]['videos']
                        paired_video = random.choice(other_videos)
                        file.write(f"1 {speaker_data[speaker_id][show_id]['path']}/{video} {speaker_data[speaker_id][other_show]['path']}/{paired_video}\n")
            
            # Generating negative pairs
            for show_id in shows:
                videos = speaker_data[speaker_id][show_id]['videos']
                other_speakers = [sp for sp in all_speakers if sp != speaker_id]
                
                for video in videos:
                    if other_speakers:
                        other_speaker = random.choice(other_speakers)
                        other_show = random.choice(list(speaker_data[other_speaker].keys()))
                        other_videos = speaker_data[other_speaker][other_show]['videos']
                        paired_video = random.choice(other_videos)
                        file.write(f"0 {speaker_data[speaker_id][show_id]['path']}/{video} {speaker_data[other_speaker][other_show]['path']}/{paired_video}\n")

# Load the saved speaker data
test_data = load_data('test_speaker_data.json')

# Combine dev and test data for more pairing options

# Generate pairs and save to file
output_file = 'voxceleb2_testpairs.txt'  # should be 72,474 pairs/lines
generate_pairs(test_data, output_file)

print(f"Pairs have been saved to {output_file}.")
