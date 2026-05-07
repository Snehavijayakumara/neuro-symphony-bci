import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Add project root so we can import from scripts/demo_pipeline.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.demo_pipeline import (  # noqa: E402
    DURATION,
    N_CHANNELS,
    SFREQ,
    bandpass_filter,
    extract_features,
    generate_synthetic_eeg,
    normalize_channels,
)


st.set_page_config(page_title='Neuro-Symphony EEG Demo', layout='wide')
st.title('Neuro-Symphony EEG Demo')


@st.cache_data
def _generate_data() -> tuple[np.ndarray, np.ndarray]:
    return generate_synthetic_eeg(N_CHANNELS, SFREQ, DURATION)


# Section 1: Generate EEG Signal
st.header('Section 1: Generate EEG Signal')
if st.button('Generate EEG Signal'):
    data, times = _generate_data()
    st.session_state['raw_data'] = data
    st.session_state['times'] = times

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, data[0], color='tab:blue', linewidth=1.0)
    ax.set_title('Raw Synthetic EEG - Channel 0')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (a.u.)')
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    st.success(f'Generated synthetic EEG with shape: {data.shape}')


# Section 2: Run Preprocessing
st.header('Section 2: Run Preprocessing')
if st.button('Run Preprocessing'):
    if 'raw_data' not in st.session_state or 'times' not in st.session_state:
        data, times = _generate_data()
        st.session_state['raw_data'] = data
        st.session_state['times'] = times
    else:
        data = st.session_state['raw_data']
        times = st.session_state['times']

    filtered = bandpass_filter(data, 1.0, 40.0, SFREQ, order=4)
    normed = normalize_channels(filtered)

    st.session_state['filtered_data'] = filtered
    st.session_state['normed_data'] = normed

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, filtered[0], label='Filtered (1-40 Hz)', alpha=0.9)
    ax.plot(times, normed[0], label='Normalized', alpha=0.9)
    ax.set_title('Processed EEG - Channel 0')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (a.u.)')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    st.success('Preprocessing completed.')


# Section 3: Extract Features
st.header('Section 3: Extract Features')
if st.button('Extract Features'):
    if 'normed_data' not in st.session_state:
        if 'raw_data' not in st.session_state:
            data, _ = _generate_data()
            st.session_state['raw_data'] = data
        else:
            data = st.session_state['raw_data']
        filtered = bandpass_filter(data, 1.0, 40.0, SFREQ, order=4)
        normed = normalize_channels(filtered)
        st.session_state['normed_data'] = normed
    else:
        normed = st.session_state['normed_data']

    feats = extract_features(normed, SFREQ)
    st.session_state['features'] = feats

    st.subheader('Feature Matrix Shape')
    st.write(feats.shape)

    st.subheader('First 3 Channels Features')
    feature_names = ['mean', 'variance', 'std', 'delta', 'theta', 'alpha', 'beta']
    df = pd.DataFrame(feats[:3], columns=feature_names)
    df.index = [f'channel_{i}' for i in range(3)]
    st.dataframe(df, use_container_width=True)

    st.success('Feature extraction completed.')
