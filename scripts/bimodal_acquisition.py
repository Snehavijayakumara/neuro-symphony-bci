import os
import numpy as np
import matplotlib.pyplot as plt


def normalize_signal(x: np.ndarray) -> np.ndarray:
    """Normalize signal to zero mean and unit variance."""
    std = np.std(x)
    if std == 0:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def generate_synthetic_eeg(n_channels: int, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate synthetic EEG signals (8-30 Hz + noise)."""
    eeg = np.zeros((n_channels, t.size), dtype=float)
    for ch in range(n_channels):
        f1 = rng.uniform(8.0, 14.0)
        f2 = rng.uniform(14.0, 30.0)
        a1 = rng.uniform(0.8, 1.5)
        a2 = rng.uniform(0.4, 1.0)
        p1 = rng.uniform(0, 2 * np.pi)
        p2 = rng.uniform(0, 2 * np.pi)

        signal = (
            a1 * np.sin(2 * np.pi * f1 * t + p1)
            + a2 * np.sin(2 * np.pi * f2 * t + p2)
        )
        noise = rng.normal(0.0, 0.5, size=t.size)
        eeg[ch] = signal + noise
    return eeg


def generate_synthetic_fnirs(n_channels: int, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate synthetic fNIRS signals (0.01-0.2 Hz + drift + noise)."""
    fnirs = np.zeros((n_channels, t.size), dtype=float)
    for ch in range(n_channels):
        f1 = rng.uniform(0.01, 0.08)
        f2 = rng.uniform(0.08, 0.2)
        a1 = rng.uniform(0.4, 1.0)
        a2 = rng.uniform(0.2, 0.6)
        p1 = rng.uniform(0, 2 * np.pi)
        p2 = rng.uniform(0, 2 * np.pi)

        slow_signal = (
            a1 * np.sin(2 * np.pi * f1 * t + p1)
            + a2 * np.sin(2 * np.pi * f2 * t + p2)
        )
        drift = 0.03 * t
        noise = rng.normal(0.0, 0.05, size=t.size)
        fnirs[ch] = slow_signal + drift + noise
    return fnirs


def main() -> None:
    # Time axis setup
    eeg_sfreq = 256
    duration = 10
    time = np.arange(0, duration, 1.0 / eeg_sfreq)

    # Synthetic bimodal signal generation on shared time axis
    rng = np.random.default_rng(42)
    eeg = generate_synthetic_eeg(n_channels=64, t=time, rng=rng)
    fnirs = generate_synthetic_fnirs(n_channels=16, t=time, rng=rng)

    # Synchronization confirmation
    print('Bimodal Data Generated')
    print('EEG shape:', eeg.shape)
    print('fNIRS shape:', fnirs.shape)

    # Visualization
    eeg_ch0 = eeg[0]
    fnirs_ch0 = fnirs[0]

    eeg_norm = normalize_signal(eeg_ch0)
    fnirs_norm = normalize_signal(fnirs_ch0)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(time, eeg_ch0, color='tab:blue', linewidth=1.0)
    axes[0].set_title('EEG Channel 0')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(alpha=0.3)

    axes[1].plot(time, fnirs_ch0, color='tab:green', linewidth=1.0)
    axes[1].set_title('fNIRS Channel 0')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(alpha=0.3)

    axes[2].plot(time, eeg_norm, label='EEG Ch0 (normalized)', color='tab:blue', linewidth=1.0)
    axes[2].plot(time, fnirs_norm, label='fNIRS Ch0 (normalized)', color='tab:orange', linewidth=1.0)
    axes[2].set_title('Overlay: EEG + fNIRS (Normalized)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Normalized amplitude')
    axes[2].legend(loc='upper right')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'bimodal_plot.png')
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print('Plot saved to:', output_path)


if __name__ == '__main__':
    main()
