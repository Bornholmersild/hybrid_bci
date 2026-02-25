import numpy as np
import scipy.stats as stats
from scipy.signal import welch
import antropy as ant

import re
from pathlib import Path
from typing import List, Union
from collections.abc import Callable

# Manage plots
from flask import json
import matplotlib.pyplot as plt

# Manage data
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# From own implementations
from src.utilities.preprocessing import Filtering, EEG_preprocessing, EMG_preprocessing, RejectBadEpochs
from src.utilities.load_and_visualize_data import load_datasets, visualize_EEG

class FeatureExtraction():
    def __init__(self, fs = 125):
        self.fs = fs

    def extract_statistical_features(self, window):
        # window shape: (channels, samples)

        feats = []

        for ch in range(window.shape[0]):
            signal = window[ch]

            feats.extend([
                np.mean(signal),
                np.median(signal),
                np.std(signal),
                np.var(signal),
                stats.skew(signal),
                stats.kurtosis(signal),
                np.ptp(signal)
            ])

        return np.array(feats)
    
    def extract_time_domain_features(self, signal):
        rms = np.sqrt(np.mean(signal**2))
        zero_crossings = ((signal[:-1] * signal[1:]) < 0).sum()
        autocorrelation = np.correlate(signal, signal, mode='full')[len(signal)-1]
        mean_abs_dev = np.mean(np.abs(signal - np.mean(signal)))
        max_val = np.max(signal)
        min_val = np.min(signal)
        signal_energy = np.sum(signal**2)
        
        features = {
            'RMS': rms,
            'Zero Crossings': zero_crossings,
            'Autocorrelation': autocorrelation,
            'Mean Absolute Deviation': mean_abs_dev,
            'Max Value': max_val,
            'Min Value': min_val,
            'Signal Energy': signal_energy
        }
        
        return features
    
    def extract_frequency_domain_features(self, signal):
        freqs, psd = welch(signal, self.fs)
        dominant_freq = freqs[np.argmax(psd)]
        total_power = np.sum(psd)
        band_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 40)])
        mean_freq = np.mean(freqs)
        median_freq = np.median(freqs)
        peak_freq = freqs[np.argmax(psd)]
        freq_variance = np.var(freqs)
        
        features = {
            'Dominant Frequency': dominant_freq,
            'Total Power': total_power,
            'Band Power (0.5-40 Hz)': band_power,
            'Mean Frequency': mean_freq,
            'Median Frequency': median_freq,
            'Peak Frequency': peak_freq,
            'Frequency Variance': freq_variance
        }
        
        return features
    
    def extract_entropy_features(self, signal):
        sample_entropy = ant.sample_entropy(signal)
        spectral_entropy = ant.spectral_entropy(signal, sf = self.fs, method='welch')
        perm_entropy = ant.perm_entropy(signal, normalize=True)
        svd_entropy = ant.svd_entropy(signal, order=3, delay=1)
        app_entropy = ant.app_entropy(signal)
        lziv_complexity = ant.lziv_complexity(signal)
        
        features = {
            'Sample Entropy': sample_entropy,
            'Spectral Entropy': spectral_entropy,
            'Permutation Entropy': perm_entropy,
            'SVD Entropy': svd_entropy,
            'Approximate Entropy': app_entropy,
            'LZiv Complexity': lziv_complexity
        }
        
        return features


def sliding_chunks(data, window_size, step_size):
    """
    data: (samples, channels)

    Returns:
        chunks: (n_windows, window_size, channels)
    """
    windows = sliding_window_view(
        data,
        window_shape=window_size,
        axis=0
    )

    return windows[::step_size]

def load_data():
    EEG_FREQ = 125
    TRIAL_PERIOD = 9

    EMG_CONFIG_DICT = {
        'rms_windowsize' : 32,
        'rms_stepsize' : 16,
        'hampel_windowsize' : 100,
        'hampel_sigma' : 2,
        'hampel_plot_option' : [False, None],
        'include_EMG' : True
    }

    REJECT_CONFIG_DICT = {
        'EEG_epoch_rejection_tolerance' : 6,
        'EMG_epoch_rejection_tolerance' : 6
    }

    base_dir = Path().resolve() / 'src/experiment/data'
    load_ins = load_datasets(base_dir = base_dir)

    EEG_files = load_ins.find_flex_files(
        subjects = 'subject_8',
        modality = 'EEG',
        fingers = 'thumb',
        prefix = 'flex'
    )

    EMG_files = load_ins.find_flex_files(
        subjects = 'subject_8',
        modality = 'EMG',
        fingers = 'thumb',
        prefix = 'flex'
    )

    marker_files = load_ins.find_flex_files(
        subjects = 'subject_8',
        modality = 'Markers',
        fingers = 'thumb',
        prefix = 'flex'
    )

    EEG_ins = EEG_preprocessing(fs = 125, bandpass_lowcut = 0.05, bandpass_highcut = 32, trial_period = 9, trim_period = 3)
    EMG_ins = EMG_preprocessing(fs = 2000, bandpass_lowcut = 20, bandpass_highcut = 450, trial_period = 9, trim_period = 3)
    reject_ins = RejectBadEpochs(base_dir = base_dir)

    SELECT_EXP = 0
    EEG, RMS, EMG, epochs_overview = load_ins.load_datasets(
        path_to_EEG_files = EEG_files[SELECT_EXP],
        path_to_EMG_files = EMG_files[SELECT_EXP],
        EEG_preprocessing_func = EEG_ins.preprocessing_routine,
        EMG_preprocessing_func = EMG_ins.preprocessing_routine,
        EMG_config_dict = EMG_CONFIG_DICT
    )

    markers = load_ins.load_datasets_marker(marker_files)[SELECT_EXP]

    reject_mask = reject_ins.reject_routine(data_file_per_finger = EEG_files,
                                            epochs_overview = epochs_overview,
                                            EEG_data = EEG,
                                            RMS_data = RMS,
                                            reject_config_dict = REJECT_CONFIG_DICT,
                                            EEG_useable_channels = None)
    
    reject_mask_indices = np.where(reject_mask)[0]

    total_epochs = sum(epochs_overview)
    EEG_epoch = EEG.reshape(total_epochs, EEG.shape[0] // total_epochs, EEG.shape[1])
    RMS_epoch = RMS.reshape(total_epochs, RMS.shape[0] // total_epochs, RMS.shape[1])
    EMG_epoch = EMG.reshape(total_epochs, EMG.shape[0] // total_epochs, EMG.shape[1])
    
    EEG_epoch_clean = EEG_epoch[~reject_mask]
    RMS_epoch_clean = RMS_epoch[~reject_mask]
    EMG_epoch_clean = EMG_epoch[~reject_mask]
    print(EEG_epoch_clean.shape, RMS_epoch_clean.shape, EMG_epoch_clean.shape)

    return EEG_epoch_clean, total_epochs

# One article
# 250 fs
# 2 sec window size
# 0.1 s between window. 2-4, 2.1-4.1, 2.2-4.2
# Filtering 8-30 Hz

# Other article
# 160 fs
# 7 segments across 2s and 50% overlap
# Filtering 0.5-70Hz

def main():
    features_ins = FeatureExtraction(fs = 125)
    window_size = 2*125
    step_size = 25
    
    EEG, total_epochs = load_data()

    all_windows = np.array([                        
        sliding_chunks(epo, window_size, step_size)
        for epo in EEG
    ])                                            # Shape (num_epochs, num_windows, channels, window_samples)

    samples = EEG.shape[0]
    print(all_windows.shape)

    print('samples:', samples)
    print(f'Remaining sampels: {((samples - window_size) / step_size) + 1}')

    features = []

    for trial in all_windows:
        trial_features = []
        for window in trial:
            feat = features_ins.extract_statistical_features(window)  # compute across channels properly
            trial_features.append(feat)
        features.append(trial_features)

    X_feat = np.array(features)                     # (num_epochs, num_windows (time steps - LSTM sequence), feature)

    print(X_feat.shape)
    
if __name__ == '__main__':
    main()

    

