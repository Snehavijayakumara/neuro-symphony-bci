import mne
import matplotlib.pyplot as plt

file_path = r"F:\ds004514_real\sub-01\nirs\sub-01_task-nirs_nirs.snirf"

print("Loading real fNIRS...")

raw = mne.io.read_raw_snirf(file_path, preload=True)

print(raw)

print("Cropping first 20 seconds...")

raw.crop(tmin=0, tmax=20)

print("Selecting first fNIRS channel...")

data, times = raw[:1]

plt.figure(figsize=(12, 4))

plt.plot(times, data[0])

plt.title("Real fNIRS Signal")

plt.xlabel("Time (s)")

plt.ylabel("Amplitude")

plt.grid(True)

plt.show()

print("Real fNIRS loaded successfully.")