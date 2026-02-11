import os
from pydub import AudioSegment
import json

# Paths
audio_file = r"C:\Users\Berken\Documents\GitHub\chatterbox\combined_voice_deessed.wav"
segments_file = r"C:\Users\Berken\Documents\GitHub\chatterbox\segments.txt"
output_dir = r"C:\Users\Berken\Documents\GitHub\chatterbox\xtts_dataset"

# Create output directories
wavs_dir = os.path.join(output_dir, "wavs")
os.makedirs(wavs_dir, exist_ok=True)

print(f"Loading audio: {audio_file}")
audio = AudioSegment.from_wav(audio_file)

print(f"Reading segments: {segments_file}")
with open(segments_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Process segments
metadata = []
for i, line in enumerate(lines):
    parts = line.strip().split("|")
    if len(parts) != 3:
        continue

    start_sec = float(parts[0])
    end_sec = float(parts[1])
    text = parts[2]

    # Skip very short or very long segments
    duration = end_sec - start_sec
    if duration < 1.0 or duration > 15.0:
        print(f"  Skipping segment {i}: duration {duration:.1f}s")
        continue

    # Skip segments with very short text
    if len(text) < 5:
        print(f"  Skipping segment {i}: text too short")
        continue

    # Extract audio segment
    start_ms = int(start_sec * 1000)
    end_ms = int(end_sec * 1000)
    segment_audio = audio[start_ms:end_ms]

    # Save as WAV
    wav_filename = f"segment_{i:04d}.wav"
    wav_path = os.path.join(wavs_dir, wav_filename)
    segment_audio.export(wav_path, format="wav")

    # Add to metadata
    metadata.append({
        "audio_file": f"wavs/{wav_filename}",
        "text": text,
        "speaker_name": "target_voice"
    })

    print(f"  Saved: {wav_filename} ({duration:.1f}s)")

# Save metadata for XTTS
metadata_file = os.path.join(output_dir, "metadata.csv")
with open(metadata_file, "w", encoding="utf-8") as f:
    for item in metadata:
        # Format: audio_file|text|speaker_name
        f.write(f"{item['audio_file']}|{item['text']}|{item['speaker_name']}\n")

print(f"\n--- Done! ---")
print(f"Total segments: {len(metadata)}")
print(f"Dataset dir: {output_dir}")
print(f"Metadata file: {metadata_file}")
