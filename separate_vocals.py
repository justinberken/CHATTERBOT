import subprocess
import os
import shutil

input_file = r"C:\Users\Berken\Music\RTX 3080 ｜ RTX 3090 ｜ RX 6900XT ｜ GPU PRODUCTIVITY BENCHMARKS!.mp3"
output_dir = r"C:\Users\Berken\Documents\GitHub\chatterbox\separated"

print("Running Demucs to isolate vocals...")
print("This may take a few minutes...")

# Run demucs with htdemucs model (best quality)
subprocess.run([
    "python", "-m", "demucs",
    "--two-stems", "vocals",  # Only separate vocals vs other
    "-o", output_dir,
    input_file
], check=True)

print(f"\nDone! Check the '{output_dir}' folder")
print("Look for the 'vocals.wav' file - that's the isolated voice")
