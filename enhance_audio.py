from resemble_enhance.enhancer.inference import denoise, enhance
import torchaudio
import torch

input_file = r"C:\Users\Berken\Music\RTX 3080 ｜ RTX 3090 ｜ RX 6900XT ｜ GPU PRODUCTIVITY BENCHMARKS!.mp3"
output_file = r"C:\Users\Berken\Documents\GitHub\chatterbox\enhanced_voice.wav"

# Use RTX 3090
device = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'
print(f"Using GPU: {torch.cuda.get_device_name(int(device[-1]))}")

print(f"Loading: {input_file}")
audio, sr = torchaudio.load(input_file)

# Convert to mono if stereo
if audio.shape[0] > 1:
    audio = audio.mean(dim=0, keepdim=True)

audio = audio.squeeze(0)

print("Enhancing audio (removing noise + reverb)...")
# denoise only, or use enhance() for full enhancement
enhanced, new_sr = enhance(audio, sr, device=device, nfe=64)

print(f"Saving to: {output_file}")
torchaudio.save(output_file, enhanced.unsqueeze(0).cpu(), new_sr)

print("Done! Use 'enhanced_voice.wav' as your voice reference.")
