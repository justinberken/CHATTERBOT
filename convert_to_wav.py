from pydub import AudioSegment
import os

# Input MP3 file
input_file = r"C:\Users\Berken\Music\RTX 3080 ｜ RTX 3090 ｜ RX 6900XT ｜ GPU PRODUCTIVITY BENCHMARKS!.mp3"

# Output WAV file
output_file = r"C:\Users\Berken\Documents\GitHub\chatterbox\voice_ref.wav"

print(f"Loading: {input_file}")
audio = AudioSegment.from_mp3(input_file)

# Convert to mono and set sample rate to 24kHz (optimal for Chatterbox)
audio = audio.set_channels(1)
audio = audio.set_frame_rate(24000)

# Get duration
duration_seconds = len(audio) / 1000
print(f"Duration: {duration_seconds:.1f} seconds")

# Using full length - no trimming
print("Using full audio length (no trimming)")

# Export as WAV
audio.export(output_file, format="wav")
print(f"Saved to: {output_file}")
