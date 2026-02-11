import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use RTX 3090

import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from TTS.api import TTS
import torchaudio

print("Loading XTTS v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# Use the de-essed combined audio as reference
reference_audio = r"C:\Users\Berken\Documents\GitHub\chatterbox\combined_voice_deessed.wav"

text = "Hello, this is a test of XTTS voice cloning. I hope this sounds natural and like the original voice without any strange accent."

print(f"Using reference: {reference_audio}")
print(f"Generating: {text}")

output_file = "xtts_output.wav"

tts.tts_to_file(
    text=text,
    file_path=output_file,
    speaker_wav=reference_audio,
    language="en"
)

print(f"Done! Saved to: {output_file}")
