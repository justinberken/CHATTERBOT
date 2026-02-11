import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use RTX 3090 only

from df.enhance import enhance, init_df, load_audio, save_audio
import torch

input_file = r"C:\Users\Berken\Music\RTX 3080 ｜ RTX 3090 ｜ RX 6900XT ｜ GPU PRODUCTIVITY BENCHMARKS!.mp3"
output_file = r"C:\Users\Berken\Documents\GitHub\chatterbox\clean_voice.wav"

print("Loading DeepFilterNet model (using RTX 3090)...")
model, df_state, _ = init_df()

print(f"Loading: {input_file}")
audio, _ = load_audio(input_file, sr=df_state.sr())

print("Removing reverb and noise...")
enhanced = enhance(model, df_state, audio)

print(f"Saving to: {output_file}")
save_audio(output_file, enhanced, df_state.sr())

print("Done! Use 'clean_voice.wav' as your voice reference.")
