from chatterbox.tts import ChatterboxTTS
import torchaudio
import torch

# Use RTX 3090 (cuda:1) - RTX 5090 not yet supported
device = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'
print(f"Using GPU: {torch.cuda.get_device_name(int(device[-1]))}")

print("Loading model...")
model = ChatterboxTTS.from_pretrained(device=device)

# Voice reference file (cleaned with DeepFilterNet)
voice_ref = "clean_voice.wav"
print(f"Using voice reference: {voice_ref}")

# Text to speak
text = "Hello, this is a test of voice cloning with Chatterbox. I hope this sounds like the original voice."

print("Generating speech with cloned voice...")
wav = model.generate(text, audio_prompt_path=voice_ref)

output_file = "cloned_output.wav"
torchaudio.save(output_file, wav, model.sr)
print(f"Done! Saved to {output_file}")
