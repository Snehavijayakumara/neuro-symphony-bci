"""
Demo EEG preprocessing and feature extraction pipeline.
Generates synthetic EEG, applies bandpass filtering, normalizes, extracts features,
prints results, and saves a plot of one channel.

Run:
python scripts/demo_pipeline.py
"""

import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


# PARAMETERS
N_CHANNELS = 64
SFREQ = 256  # Hz
DURATION = 10  # seconds
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
os.makedirs(OUT_DIR, exist_ok=True)


def generate_synthetic_eeg(n_channels: int, sfreq: int, duration: float) -> (np.ndarray, np.ndarray):
    """Generate synthetic EEG: mixture of sine waves + noise.

    Returns:
        data: ndarray (n_channels, n_samples)
        times: ndarray (n_samples,)
    """
    n_samples = int(sfreq * duration)
    times = np.arange(n_samples) / sfreq

    base_freqs = [6.0, 10.0, 20.0]  # theta, alpha, beta components
    data = np.zeros((n_channels, n_samples), dtype=float)

    rng = np.random.default_rng(42)
    for ch in range(n_channels):
        signal_comb = np.zeros(n_samples, dtype=float)
        for bf in base_freqs:
            amp = 5.0 * (0.5 + rng.random())  # channel-varying amplitude
            phase = rng.random() * 2 * np.pi
            signal_comb += amp * np.sin(2 * np.pi * bf * times + phase)
        # add low-frequency drift
        signal_comb += 0.5 * np.sin(2 * np.pi * 0.5 * times + rng.random())
        # add Gaussian noise
        signal_comb += rng.normal(0, 2.0, size=n_samples)
        data[ch] = signal_comb

    return data, times


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    """Apply zero-phase bandpass filter to multi-channel data (ch, samples)."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # apply along last axis (samples)
    filtered = signal.filtfilt(b, a, data, axis=-1)
    return filtered


def normalize_channels(data: np.ndarray) -> np.ndarray:
    """Z-score normalization per channel."""
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (data - mean) / std


def band_power(psd_freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    """Compute band power by integrating PSD between fmin and fmax."""
    mask = (psd_freqs >= fmin) & (psd_freqs < fmax)
    # manual trapezoidal integration to avoid backend issues
    if not np.any(mask):
        return 0.0
    x = psd_freqs[mask]
    y = psd[mask]
    if x.size < 2:
        return 0.0
    return float(np.sum((y[:-1] + y[1:]) * np.diff(x) / 2.0))


def extract_features(data: np.ndarray, fs: float) -> np.ndarray:
    """Extract features per channel: mean, var, std, band powers (delta, theta, alpha, beta).

    Returns:
        features: ndarray (n_channels, 7)
    """
    n_channels = data.shape[0]
    features = np.zeros((n_channels, 7), dtype=float)

    # compute PSD with Welch
    nperseg = 512
    for ch in range(n_channels):
        x = data[ch]
        mean = np.mean(x)
        var = np.var(x)
        stdv = np.std(x)

        freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg)

        bp_delta = band_power(freqs, psd, 1.0, 4.0)
        bp_theta = band_power(freqs, psd, 4.0, 8.0)
        bp_alpha = band_power(freqs, psd, 8.0, 13.0)
        bp_beta = band_power(freqs, psd, 13.0, 30.0)

        features[ch] = [mean, var, stdv, bp_delta, bp_theta, bp_alpha, bp_beta]

    return features


def main():
    # 1) Generate synthetic EEG
    data, times = generate_synthetic_eeg(N_CHANNELS, SFREQ, DURATION)
    print('Generated synthetic EEG with shape:', data.shape)

    # 2) Preprocessing: bandpass 1-40 Hz
    filtered = bandpass_filter(data, 1.0, 40.0, SFREQ, order=4)

    # 3) Normalize per channel
    normed = normalize_channels(filtered)

    # 4) Feature extraction
    feats = extract_features(normed, SFREQ)
    print('Feature Matrix Shape:')
    print(feats.shape)
    print()
    print('Sample Features for Channel 0:')
    print(f'({feats[0,0]:.6f}, {feats[0,1]:.6f}, {feats[0,2]:.6f}, {feats[0,3]:.6f}, {feats[0,4]:.6f}, {feats[0,5]:.6f}, {feats[0,6]:.6f})')
    print()
    print('First 3 Channels Feature Table:')
    print('Ch | Mean      | Var       | Std       | Delta     | Theta     | Alpha     | Beta')
    print('---+-----------+-----------+-----------+-----------+-----------+-----------+-----------')
    for i in range(min(3, feats.shape[0])):
        print(
            f'{i:>2} | '
            f'{feats[i,0]:>9.6f} | {feats[i,1]:>9.6f} | {feats[i,2]:>9.6f} | '
            f'{feats[i,3]:>9.6f} | {feats[i,4]:>9.6f} | {feats[i,5]:>9.6f} | {feats[i,6]:>9.6f}'
        )

    # 5) Plot one channel and save
    ch_to_plot = 0
    plt.figure(figsize=(10, 4))
    plt.plot(times, data[ch_to_plot], label='raw', alpha=0.6)
    plt.plot(times, filtered[ch_to_plot], label='filtered (1-40 Hz)', alpha=0.9)
    plt.plot(times, normed[ch_to_plot], label='normalized', alpha=0.9)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (a.u.)')
    plt.title(f'Channel {ch_to_plot} - synthetic EEG')
    plt.legend(loc='upper right')
    out_path = os.path.join(OUT_DIR, 'demo_channel_plot.png')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print('Saved channel plot to:', out_path)


if __name__ == '__main__':
    main()
