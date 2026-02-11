from pydub import AudioSegment, effects
import noisereduce as nr
import numpy as np
import os

input_dir = r"C:\Users\Berken\Documents\GitHub\chatterbox\voice_wavs"
output_dir = r"C:\Users\Berken\Documents\GitHub\chatterbox\voice_wavs_clean"
os.makedirs(output_dir, exist_ok=True)

# Get all WAV files
wav_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.wav')])

print(f"Found {len(wav_files)} WAV files to process\n")

for i, filename in enumerate(wav_files):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    print(f"[{i+1}/{len(wav_files)}] Processing: {filename}")

    # Load audio
    audio = AudioSegment.from_wav(input_path)
    sample_rate = audio.frame_rate

    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    # Step 1: Noise reduction
    print("    Reducing noise...")
    reduced = nr.reduce_noise(y=samples, sr=sample_rate, prop_decrease=0.8)

    # Convert back to int16
    reduced_int = np.clip(reduced, -32768, 32767).astype(np.int16)

    # Create new AudioSegment
    cleaned = AudioSegment(
        reduced_int.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )

    # Step 2: Normalize volume
    print("    Normalizing volume...")
    normalized = effects.normalize(cleaned, headroom=1.0)

    # Save
    normalized.export(output_path, format="wav")

    duration = len(normalized) / 1000 / 60
    print(f"    Saved: {filename} ({duration:.1f} min)\n")

print("--- Done ---")
print(f"Cleaned files saved to: {output_dir}")
