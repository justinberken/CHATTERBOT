from pydub import AudioSegment
import numpy as np
from scipy import signal
import os

input_file = r"C:\Users\Berken\Documents\GitHub\chatterbox\combined_voice.wav"
output_file = r"C:\Users\Berken\Documents\GitHub\chatterbox\combined_voice_deessed.wav"

print(f"Loading: {input_file}")
audio = AudioSegment.from_wav(input_file)
sample_rate = audio.frame_rate

# Convert to numpy
samples = np.array(audio.get_array_of_samples()).astype(np.float32)

print("Applying de-esser...")

# Design a bandpass filter for sibilance detection (4-9 kHz)
low_freq = 4000
high_freq = 9000
nyquist = sample_rate / 2
low = low_freq / nyquist
high = min(high_freq / nyquist, 0.99)

# Create bandpass filter to detect sibilance
b_detect, a_detect = signal.butter(4, [low, high], btype='band')

# Get the sibilant frequencies
sibilant = signal.filtfilt(b_detect, a_detect, samples)

# Calculate envelope of sibilant content
envelope = np.abs(signal.hilbert(sibilant))

# Smooth the envelope
window_size = int(sample_rate * 0.01)  # 10ms window
envelope_smooth = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')

# Calculate threshold (dynamic based on signal)
threshold = np.percentile(envelope_smooth, 85)

# Create gain reduction curve
reduction_amount = 0.5  # Reduce sibilance by 50%
gain = np.ones_like(samples)
mask = envelope_smooth > threshold
gain[mask] = 1.0 - (reduction_amount * (envelope_smooth[mask] - threshold) / (envelope_smooth[mask] + 1e-10))
gain = np.clip(gain, 1.0 - reduction_amount, 1.0)

# Smooth the gain to avoid artifacts
gain_smooth = np.convolve(gain, np.ones(window_size)/window_size, mode='same')

# Apply gain reduction only to high frequencies
# Split into low and high bands
b_high, a_high = signal.butter(4, low, btype='high')
b_low, a_low = signal.butter(4, low, btype='low')

high_freq_content = signal.filtfilt(b_high, a_high, samples)
low_freq_content = signal.filtfilt(b_low, a_low, samples)

# Apply de-essing to high frequencies only
high_freq_deessed = high_freq_content * gain_smooth

# Recombine
output = low_freq_content + high_freq_deessed

# Normalize to prevent clipping
output = output / np.max(np.abs(output)) * 32767 * 0.95

# Convert back to int16
output_int = output.astype(np.int16)

# Create AudioSegment
deessed = AudioSegment(
    output_int.tobytes(),
    frame_rate=sample_rate,
    sample_width=2,
    channels=1
)

print(f"Saving to: {output_file}")
deessed.export(output_file, format="wav")

duration = len(deessed) / 1000 / 60
print(f"Done! Duration: {duration:.1f} minutes")
