from chatterbox.tts import ChatterboxTTS
import torchaudio
import torch

# Use RTX 3090
device = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'
print(f"Using GPU: {torch.cuda.get_device_name(int(device[-1]))}")

print("Loading model...")
model = ChatterboxTTS.from_pretrained(device=device)

voice_ref = "voice_ref.wav"
text = "Hello, this is a test of voice cloning with Chatterbox. I hope this sounds natural without any accent."

# Test different exaggeration levels
exaggeration_levels = [0.3, 0.5, 0.7]

for exag in exaggeration_levels:
    print(f"\nGenerating with exaggeration={exag}...")
    wav = model.generate(
        text,
        audio_prompt_path=voice_ref,
        exaggeration=exag
    )
    output_file = f"output_exag_{exag}.wav"
    torchaudio.save(output_file, wav, model.sr)
    print(f"Saved to {output_file}")

print("\n--- Done! Compare the files: ---")
print("output_exag_0.3.wav - Less voice copying, more neutral")
print("output_exag_0.5.wav - Balanced (default)")
print("output_exag_0.7.wav - More voice copying")
