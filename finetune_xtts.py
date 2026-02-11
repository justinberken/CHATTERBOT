import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use RTX 3090

import torch

# Fix PyTorch 2.6 weights_only issue
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.config.shared_configs import BaseDatasetConfig

# Paths
dataset_dir = r"C:\Users\Berken\Documents\GitHub\chatterbox\xtts_dataset"
output_dir = r"C:\Users\Berken\Documents\GitHub\chatterbox\xtts_finetuned"
os.makedirs(output_dir, exist_ok=True)

# Find the model directory
model_dir = os.path.join(
    os.path.expanduser("~"),
    "AppData", "Local", "tts",
    "tts_models--multilingual--multi-dataset--xtts_v2"
)

print("Setting up XTTS fine-tuning...")
print(f"Dataset: {dataset_dir}")
print(f"Model dir: {model_dir}")
print(f"Output: {output_dir}")

# Load config
config = XttsConfig()
config.load_json(os.path.join(model_dir, "config.json"))

# Update for training
config.output_path = output_dir
config.run_name = "xtts_finetune"
config.epochs = 10
config.batch_size = 1  # Smaller batch for memory
config.eval_batch_size = 1
config.lr = 5e-6
config.print_step = 25
config.save_step = 500

# Initialize model
print("\nLoading XTTS model...")
model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_dir=model_dir,
    eval=False
)
model.cuda()

print(f"Model loaded on: {next(model.parameters()).device}")

# Load training data
print("\nLoading training data...")
import csv
from pathlib import Path

wavs_dir = os.path.join(dataset_dir, "wavs")
metadata_file = os.path.join(dataset_dir, "metadata.csv")

samples = []
with open(metadata_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) >= 2:
            audio_path = os.path.join(dataset_dir, parts[0])
            text = parts[1]
            if os.path.exists(audio_path):
                samples.append({
                    "audio_file": audio_path,
                    "text": text,
                    "language": "en"
                })

print(f"Found {len(samples)} training samples")

# Fine-tuning loop
from torch.optim import AdamW
from tqdm import tqdm
import torchaudio

optimizer = AdamW(model.parameters(), lr=config.lr)

print("\n--- Starting fine-tuning ---")
print(f"Epochs: {config.epochs}")
print(f"Samples: {len(samples)}")

for epoch in range(config.epochs):
    model.train()
    total_loss = 0

    progress = tqdm(samples, desc=f"Epoch {epoch+1}/{config.epochs}")

    for sample in progress:
        try:
            # Load audio
            waveform, sr = torchaudio.load(sample["audio_file"])
            if sr != 22050:
                waveform = torchaudio.functional.resample(waveform, sr, 22050)
            waveform = waveform.mean(dim=0)  # Mono

            # Get speaker embedding from audio
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                audio_path=sample["audio_file"]
            )

            # Forward pass
            optimizer.zero_grad()

            outputs = model(
                text_tokens=None,
                text=sample["text"],
                language="en",
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                wav=waveform.unsqueeze(0).cuda()
            )

            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                progress.set_postfix(loss=loss.item())

        except Exception as e:
            print(f"\nError processing {sample['audio_file']}: {e}")
            continue

    avg_loss = total_loss / len(samples) if samples else 0
    print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

    # Save checkpoint
    if (epoch + 1) % 2 == 0:
        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

# Save final model
final_path = os.path.join(output_dir, "model_final.pth")
torch.save(model.state_dict(), final_path)
print(f"\n--- Done! ---")
print(f"Final model saved to: {final_path}")
