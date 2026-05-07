import os
from typing import Optional
import mne
import pandas as pd


class DataLoader:
    """Simple dataset loader for ds004514 referenced at data/raw/ds004514."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)

    def subject_eeg_bdf(self, subject: str = 'sub-01') -> str:
        path = os.path.join(self.root, subject, 'eeg', f"{subject}_task-eeg_eeg.bdf")
        return path

    def load_raw_bdf(self, subject: str = 'sub-01', preload: bool = True) -> mne.io.BaseRaw:
        path = self.subject_eeg_bdf(subject)
        if not os.path.exists(path):
            raise FileNotFoundError(f'BDF file not found: {path}')
        # Try reading with MNE BDF reader first
        try:
            raw = mne.io.read_raw_bdf(path, preload=preload, verbose=False)
            return raw
        except Exception as e:
            print('read_raw_bdf failed:', e)

        # Try using pyedflib to read BDF and convert to MNE RawArray
        try:
            import numpy as np
            import pyedflib

            with pyedflib.EdfReader(path) as f:
                n_signals = f.signals_in_file
                ch_names = f.getSignalLabels()
                nsamples = f.getNSamples()[0]
                sfreqs = f.getSampleFrequencies()
                sfreq = float(sfreqs[0]) if len(sfreqs) > 0 else float(f.getSampleFrequencies(0))

                data = np.vstack([f.readSignal(i) for i in range(n_signals)])

            info = mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types=['eeg'] * len(ch_names))
            raw = mne.io.RawArray(data, info)
            return raw
        except Exception as e_py:
            raise RuntimeError(f'Could not read BDF file with pyedflib: {e_py}')

    def read_tsv(self, subject: str = 'sub-01', filename: str = 'channels.tsv') -> pd.DataFrame:
        path = os.path.join(self.root, subject, 'eeg', filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f'TSV file not found: {path}')
        return pd.read_csv(path, sep='\t')

    def print_dataset_info(self, subject: str = 'sub-01') -> None:
        print('DATA ROOT:', self.root)
        eeg_dir = os.path.join(self.root, subject, 'eeg')
        print('Eeg dir exists:', os.path.exists(eeg_dir))
        files = []
        if os.path.exists(eeg_dir):
            files = os.listdir(eeg_dir)
        print('Files:', files)

        try:
            raw = self.load_raw_bdf(subject)
            print(raw.info)
        except Exception as e:
            print('Could not read raw:', e)
