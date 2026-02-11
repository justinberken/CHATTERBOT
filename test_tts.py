from chatterbox.tts import ChatterboxTTS
import torchaudio
import torch

# Check if CUDA is available
# RTX 5090 (cuda:0) is too new, use RTX 3090 (cuda:1) instead
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    device = 'cuda:1'
    print(f"Using GPU: {torch.cuda.get_device_name(1)}")
elif torch.cuda.is_available():
    device = 'cuda:0'
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("CUDA not available, using CPU (slower)")

print("Loading model...")
model = ChatterboxTTS.from_pretrained(device=device)

print("Generating speech...")
wav = model.generate("Hello, this is a test of Chatterbox text to speech.")

print("Saving to test_output.wav...")
torchaudio.save("test_output.wav", wav, model.sr)

print("Done! Play test_output.wav to hear the result.")
