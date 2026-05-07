import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.data_loader import DataLoader
import mne


def preprocess(subject: str = 'sub-01'):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ds_root = os.path.join(project_root, 'data', 'raw', 'ds004514')
    out_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(out_dir, exist_ok=True)

    loader = DataLoader(ds_root)
    raw = loader.load_raw_bdf(subject)

    # Bandpass 1-40 Hz
    raw.filter(1.0, 40.0, fir_design='firwin')

    # Notch filter at 50 Hz
    raw.notch_filter(freqs=[50.0])

    # Set average reference
    raw.set_eeg_reference('average')

    # Extract events from annotations if present
    try:
        events, event_id = mne.events_from_annotations(raw)
        print('Found events via annotations:', event_id)
    except Exception as e:
        print('No annotations/events found:', e)
        events = None
        event_id = None

    # Create epochs if events available
    if events is not None and event_id is not None and len(event_id) > 0:
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.2, tmax=0.8, baseline=(None, 0), preload=True)
        epochs_fname = os.path.join(out_dir, f'{subject}_epochs-epo.fif')
        epochs.save(epochs_fname, overwrite=True)
        print('Saved epochs to', epochs_fname)

    # Save cleaned raw
    raw_fname = os.path.join(out_dir, f'{subject}_cleaned_raw.fif')
    raw.save(raw_fname, overwrite=True)
    print('Saved cleaned raw to', raw_fname)


if __name__ == '__main__':
    preprocess('sub-01')
