from pydub import AudioSegment
import os

# All input video files
input_files = [
    r"C:\Users\Berken\Downloads\DOWNLOADS 2025\DOWNLOADS BACKUP\PARAPLANE RENDER SCENES - VIDEO WALKTHROUGH.mp4",
    r"E:\Users\JPBIV\Downloads\ELEVATION RENDER WALKTHROUGH.MP4",
    r"E:\Users\JPBIV\Desktop\DESKTOP 011923\DEFINITION WALKTHROUGH - GH INTERSECTIONS AND INFLATED SURFACE .mp4",
    r"C:\Users\Berken\Downloads\DOWNLOADS 2025\DOWNLOADS BACKUP\MAYO UNROLL\LESSON 034 V2 - VIDEO WALKTHROUGH.mkv",
    r"E:\Users\JPBIV\Downloads\MAIN STAGE v2\Model walkthrough video.mp4",
    r"C:\Users\Berken\Documents\GitHub\TONE-SPLITTER\GIT-IGNORE\TONE SPLITTER WALKTHROUGH.mkv",
    r"C:\Users\Berken\Videos\MEDIA CENTER S-WALL WALKTHROUGH.mkv",
    r"F:\10 FOOT SECTION WALKTHROUGH VIDEO.mov",
]

output_dir = r"C:\Users\Berken\Documents\GitHub\chatterbox\voice_wavs"
os.makedirs(output_dir, exist_ok=True)

total_duration = 0

for i, file_path in enumerate(input_files):
    print(f"[{i+1}/{len(input_files)}] Processing: {os.path.basename(file_path)}")

    if not os.path.exists(file_path):
        print(f"    WARNING: File not found, skipping!")
        continue

    try:
        # Load audio (pydub auto-detects format)
        audio = AudioSegment.from_file(file_path)

        # Convert to mono, 24kHz
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(24000)

        duration = len(audio) / 1000 / 60  # minutes
        print(f"    Duration: {duration:.1f} min")
        total_duration += duration

        # Save as WAV
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}.wav")
        audio.export(output_file, format="wav")
        print(f"    Saved: {output_file}")

    except Exception as e:
        print(f"    ERROR: {e}")
        continue

print(f"\n--- Done ---")
print(f"Total duration: {total_duration:.1f} minutes")
print(f"WAV files saved to: {output_dir}")
