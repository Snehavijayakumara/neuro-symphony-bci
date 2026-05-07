import numpy as np
import matplotlib.pyplot as plt
import mne

# =====================================================
# REAL EEG LOADING
# =====================================================

eeg_file = r"F:\ds004514_real\sub-01\eeg\sub-01_task-eeg_eeg.bdf"

print("Loading REAL EEG...")

raw_eeg = mne.io.read_raw_bdf(
    eeg_file,
    preload=False
)

# Crop first 20 seconds only
raw_eeg.crop(tmin=0, tmax=20)

# Load cropped EEG
raw_eeg.load_data()

# Downsample for faster processing
raw_eeg.resample(256)

print(raw_eeg)

# Extract first EEG channel
eeg_data, eeg_times = raw_eeg[:1]

# Normalize EEG
eeg_signal = eeg_data[0]
eeg_signal = eeg_signal / np.max(np.abs(eeg_signal))

# =====================================================
# REAL fNIRS LOADING
# =====================================================

fnirs_file = r"F:\ds004514_real\sub-01\nirs\sub-01_task-nirs_nirs.snirf"

print("Loading REAL fNIRS...")

raw_fnirs = mne.io.read_raw_snirf(
    fnirs_file,
    preload=True
)

# Crop first 20 seconds
raw_fnirs.crop(tmin=0, tmax=20)

print(raw_fnirs)

# Extract first fNIRS channel
fnirs_data, fnirs_times = raw_fnirs[:1]

# Normalize fNIRS
fnirs_signal = fnirs_data[0]
fnirs_signal = fnirs_signal / np.max(np.abs(fnirs_signal))

# =====================================================
# TIME ALIGNMENT
# =====================================================

min_len = min(len(eeg_times), len(fnirs_times))

eeg_times = eeg_times[:min_len]
eeg_signal = eeg_signal[:min_len]

fnirs_signal = fnirs_signal[:min_len]

# =====================================================
# BIMODAL VISUALIZATION
# =====================================================

plt.figure(figsize=(14, 6))

plt.plot(
    eeg_times,
    eeg_signal,
    label="REAL EEG",
    linewidth=1
)

plt.plot(
    eeg_times,
    fnirs_signal,
    label="REAL fNIRS",
    linewidth=2
)

plt.title("REAL Bimodal Signal Acquisition")

plt.xlabel("Time (s)")

plt.ylabel("Normalized Amplitude")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

print("REAL bimodal acquisition completed successfully.")