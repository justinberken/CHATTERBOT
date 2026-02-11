from pydub import AudioSegment
import os

input_dir = r"C:\Users\Berken\Documents\GitHub\chatterbox\voice_wavs_clean"
output_file = r"C:\Users\Berken\Documents\GitHub\chatterbox\combined_voice.wav"

# Get all WAV files
wav_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.wav')])

print(f"Found {len(wav_files)} WAV files:")
for f in wav_files:
    print(f"  - {f}")

combined = AudioSegment.empty()
total_duration = 0

for i, filename in enumerate(wav_files):
    file_path = os.path.join(input_dir, filename)
    print(f"\n[{i+1}/{len(wav_files)}] Adding: {filename}")

    audio = AudioSegment.from_wav(file_path)
    duration = len(audio) / 1000 / 60
    print(f"    Duration: {duration:.1f} min")

    combined += audio
    total_duration += duration

print(f"\n--- Stitching complete ---")
print(f"Total duration: {total_duration:.1f} minutes")

print(f"Saving to: {output_file}")
combined.export(output_file, format="wav")

print("Done!")
