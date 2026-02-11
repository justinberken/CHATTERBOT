import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use RTX 3090

from chatterbox.tts import ChatterboxTTS
import torchaudio
import torch

print("Using GPU: RTX 3090")

print("Loading model...")
model = ChatterboxTTS.from_pretrained(device='cuda')

voice_ref = "clean_voice.wav"
text = "Hello, this is a test of voice cloning with Chatterbox. I hope this sounds natural and clear without any strange accent."

# Different parameter combinations to compare
configs = [
    {"exaggeration": 0.2, "cfg_weight": 0.5},
    {"exaggeration": 0.3, "cfg_weight": 0.5},
    {"exaggeration": 0.5, "cfg_weight": 0.5},  # default
    {"exaggeration": 0.7, "cfg_weight": 0.5},
    {"exaggeration": 0.5, "cfg_weight": 0.3},
    {"exaggeration": 0.5, "cfg_weight": 0.7},
]

print(f"\nUsing voice reference: {voice_ref}")
print(f"Text: {text}\n")

for i, cfg in enumerate(configs):
    exag = cfg["exaggeration"]
    cfg_w = cfg["cfg_weight"]

    print(f"[{i+1}/{len(configs)}] Generating: exag={exag}, cfg={cfg_w}...")

    wav = model.generate(
        text,
        audio_prompt_path=voice_ref,
        exaggeration=exag,
        cfg_weight=cfg_w
    )

    output_file = f"compare_{i+1}_exag{exag}_cfg{cfg_w}.wav"
    torchaudio.save(output_file, wav, model.sr)
    print(f"    Saved: {output_file}")

print("\n--- Done! Compare these files: ---")
for i, cfg in enumerate(configs):
    print(f"compare_{i+1}_exag{cfg['exaggeration']}_cfg{cfg['cfg_weight']}.wav")
