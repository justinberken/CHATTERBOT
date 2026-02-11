import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use RTX 3090

import whisper
import torch

input_file = r"C:\Users\Berken\Documents\GitHub\chatterbox\combined_voice_deessed.wav"
output_file = r"C:\Users\Berken\Documents\GitHub\chatterbox\transcript.txt"

print("Loading Whisper model (large-v3 for best accuracy)...")
model = whisper.load_model("large-v3", device="cuda")

print(f"Transcribing: {input_file}")
print("This may take a few minutes...")

result = model.transcribe(input_file, language="en", verbose=True)

# Save full transcript
with open(output_file, "w", encoding="utf-8") as f:
    f.write(result["text"])

# Save segments with timestamps (needed for fine-tuning)
segments_file = r"C:\Users\Berken\Documents\GitHub\chatterbox\segments.txt"
with open(segments_file, "w", encoding="utf-8") as f:
    for seg in result["segments"]:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()
        f.write(f"{start:.2f}|{end:.2f}|{text}\n")

print(f"\nDone!")
print(f"Full transcript: {output_file}")
print(f"Segments with timestamps: {segments_file}")
