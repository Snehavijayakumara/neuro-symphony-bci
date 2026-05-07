import mne

file_path = r"F:\ds004514_real\sub-01\eeg\sub-01_task-eeg_eeg.bdf"

print("Opening metadata...")

raw = mne.io.read_raw_bdf(file_path, preload=False)

print(raw)

print("Selecting first 10 seconds...")

raw.crop(tmin=0, tmax=10)

print("Loading cropped EEG...")

raw.load_data()

print("Downsampling to 256 Hz...")

raw.resample(256)

print("EEG loaded successfully!")

print(raw.info)