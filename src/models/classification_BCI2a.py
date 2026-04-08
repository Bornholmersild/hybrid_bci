# Classification
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import sherpa

# Manage datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Manage utils
import os 
from pathlib import Path
from datetime import datetime
import logging                  # Avoid loggings from GP
from copy import deepcopy       # Used for copy model_state
from typing import Dict, List
from scipy.signal import welch

# Feature extraction
import scipy.stats as stats
from typing import Callable
from sklearn.preprocessing import StandardScaler
import antropy as ant

# Own implementations
from src.utilities.preprocessing import EEG_preprocessing, EMG_preprocessing, RejectBadEpochs, Filtering #E402
from src.utilities.trainer_and_evaluator import SingleNet_train_eval
from src.utilities.load_and_visualize_data import load_datasets, visualize_EEG
from src.models.classification_pipeline import ExperimentLogger, SingleManageDataset, SingleNetHandler #SingleNet, SingleNet_CNN, SingleManageDataset, ExperimentLogger, SingleManageDataset, build_optimizer, inspect_model

# Avoid messages for sherpa
logging.getLogger("GP").setLevel(logging.CRITICAL)
logging.getLogger("GPy").setLevel(logging.CRITICAL)

SEED = 42

class Manage2Split:
    '''
    Functionality for splitting continous data into train-validation-test split and\n
    segment trial into rest, contract and release data
    '''
    def __init__(self, seed : int):
        '''
        Parameter
        ---------
        seed : int
            For np.random generator
        '''
        self.rng = np.random.default_rng(seed)

    def build_split(self, 
                    epoch_class1 : np.ndarray,
                    epoch_class2 : np.ndarray,
                    epoch_class3 : np.ndarray, 
                    epoch_class4 : np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
        '''
        Provides a dataset and labels with 5 classes for the train-validation-test split

        Parameters
        ----------
        epoch_index : np.ndarray
            Epoch data for either EEG or EMG for index motion
        
        epoch_thumb : np.ndarray
            Epoch data for either EEG or EMG for thumb motion

        index_trials_indices : list
            Indices for a specfic split (index). Provided by split_trials function

        thumb_trials_indices : list
            Indices for a specfic split (thumb). Provided by split_trials function
        
        fs : int
            Sampling frequency

        Returns
        ----------
        X : np.ndarray
            Dataset split for either train, validation or test with all 5 classes
        
        y : np.ndarray
            Corresponding labels
        '''
        
        
        # Build features
        X = np.concatenate([
            epoch_class1,
            epoch_class2,
            epoch_class3,
            epoch_class4
        ])

        # Build labels
        y = np.concatenate([
            np.full(len(epoch_class1), 0),                    # 0 right hand
            np.full(len(epoch_class2), 1),                      # 1 left hand
            np.full(len(epoch_class3), 2),
            np.full(len(epoch_class4), 3)
        ])

        print('\n-------------------------\n'
              f'Concat data shape : {X.shape}\n'
              f'Concat label shape : {y.shape}\n')

        return X, y

    def split_trials(self, num_epochs : int, train_ratio : int = 0.8) -> tuple[list, list]:
        '''
        Provide indices for a train-validation-test split. Used per finger motion

        Parameters
        ----------
        num_trials : int
            Number of trials per finger motion

        train_ratio : int = 0.8
            Percent ratio for train split
        
        Returns
        ----------
        train_idx : list
            Indices for train split 
        
        val_idx : list
            Indices for validation split
        '''
        indices = self.rng.permutation(num_epochs)

        n_train = int(train_ratio * len(indices))

        train_idx = indices[:n_train]
        val_idx   = indices[n_train:]

        print(f'Train split indicies {train_idx.shape}\n',
              f'Valdiation split indicies {val_idx.shape}\n')

        return train_idx, val_idx

    def shuffle_BCI2a_test(self, num_epochs : int):
        return self.rng.permutation(num_epochs)

class BCI2a_preprocess():
    def __init__(self, fs, lowcut, highcut, reject_config_dict, samples_per_trial = 1001, EEG_useable_channels = None):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.samples_per_trial = samples_per_trial
        self.reject_config_dict = reject_config_dict
        self.EEG_useable_channels = EEG_useable_channels

    def detect_bad_epochs_ptp(self, data : np.ndarray, tolerance : int = 6, bad_ch_acceptance = 1) -> np.ndarray:
        """
        Detect bad epochs based on peak-to-peak amplitude.
        
        Parameters
        -----------
        data : np.ndarray 
            Epoch data with shape (n_epochs, n_times, n_channels)
        tolerance : int 
            Threshold multiplier for MAD. Higher k allows more tolerance

        Returns
        -----------
        bad_mask : np.ndarray of bools 
            A boolean mask with shape (n_epochs,). True for bad epochs, False for good epochs.
        """
        # Peak-to-peak per epoch per channel
        ptp = np.ptp(data, axis=1)  # (epochs, channels)

        # Robust threshold per channel
        med = np.median(ptp, axis = 0)                    # (c_channels,)
        mad = np.median(np.abs(ptp - med), axis = 0)      # (n_channels,) -> Purpose: Median Absolute Deviation (MAD) is a robust measure of variability that is less sensitive to outliers than standard deviation. It is calculated as the median of the absolute deviations from the median of the data. In this context, it provides a robust estimate of the variability in peak-to-peak values across epochs for each channel, which can be used to set a threshold for identifying bad epochs.

        # Avoid zero MAD
        mad[mad == 0] = 1e-12

        # Define threshold for for each channels
        threshold = med + tolerance * mad

        # Epoch is bad if ANY channel exceeds its threshold. 
        bad_mask = np.any(ptp > threshold, axis=1)

        bad_indices = np.where(bad_mask)[0]

        for idx in bad_indices:
            bad_ch_sum = np.where(ptp[idx] > threshold)[0]
            bad_ch_sum = bad_ch_sum.shape[0]
            # print(f"Epoch {idx} rejected due to channels {np.where(ptp[idx] > threshold)[0].tolist()}")
            
            if bad_ch_sum <= bad_ch_acceptance:
                bad_mask[idx] = False

        return bad_mask
    
    def reject_routine(self,
                       EEG_epoch : np.ndarray | None = None
                       ) -> np.ndarray:
        '''
        Automatic and manual rejection of bad epochs. \n
        Parameters
        ----------
        EEG_epoch : np.ndarray
            epoch EEG data after preprocessing. Shape (epochs, samples, channels)

        reject_config_dict : dict
            Dictionary with parameters for the rejection routine.\n
            Should include keys:\n
            'EEG_epoch_rejection_tolerance' -> peak-to-peak auto-rejection tolerance.\n
            'EMG_epoch_rejection_tolerance' -> peak-to-peak auto-rejection tolerance.

        EEG_useable_channels : list | None = None
            List of ints, indicating which channels are of interest. 
            NOTE: Bad epochs at other channels are not considered in the final decision

        Returns
        ----------
        all_rejections_masks : np.ndarray of bools
            Boolean mask with shape (total_epochs,). True for bad epochs, False for good epochs.
        '''
        if self.EEG_useable_channels is None:
            pass
        elif isinstance(self.EEG_useable_channels, list):
            EEG_epoch = EEG_epoch[:, :, self.EEG_useable_channels].copy()         # Extract data from selected channels
        else:
            raise ValueError(f'EEG_useable_channels is not of type list nor None. {type(self.EEG_useable_channels)}')
        
        num_epochs = EEG_epoch.shape[0]
        EEG_tolerance = self.reject_config_dict['EEG_epoch_rejection_tolerance']
        EEG_bad_ch_acceptance = self.reject_config_dict['EEG_ch_acceptance']
                
        #print(f"\nFile: {file}")
        EEG_autoreject = np.zeros(num_epochs, dtype=bool)   # Create np.ndarray of default false values corresponding to num_epochs size
        
        EEG_autoreject = self.detect_bad_epochs_ptp(EEG_epoch, tolerance = EEG_tolerance, bad_ch_acceptance = EEG_bad_ch_acceptance)    

        print('\n=====FUNC : reject_routine =====\n')
        print(f'Final combined bad epoch indicies: {np.where(EEG_autoreject)[0]}')
        print(f'Total bad epochs = {np.sum(EEG_autoreject)} out of {num_epochs}')

        return EEG_autoreject

    def preprocess_BCI2a_dataset(self, data : np.ndarray):
        # Filter
        EEG_filter_ins = Filtering(fs = self.fs)
            
        EEG_notch = EEG_filter_ins.notch(data = data, cutoff = 50, Q = 30)
        EEG_bandpass, _ = EEG_filter_ins.butter_bandpass(data = EEG_notch, lowcut = self.lowcut, highcut = self.highcut, order = 4)

        # Divide into epochs
        num_epochs = EEG_bandpass.shape[0] // self.samples_per_trial
        EEG_epoch = EEG_bandpass.reshape(num_epochs, self.samples_per_trial, EEG_bandpass.shape[1])          # (epochs, samples, channels)

        print(f'Num epochs {EEG_bandpass.shape[0] / self.samples_per_trial}',
              f'EEG epoch shape: {EEG_epoch.shape}')

        # Reject bad epochs
        reject_mask = self.reject_routine(EEG_epoch = EEG_epoch)

        EEG_epoch_clean = EEG_epoch[~reject_mask]

        EEG_car = EEG_epoch_clean - np.mean(EEG_epoch_clean, axis = 2, keepdims = True)

        EEG_epoch_norm = EEG_filter_ins.zscore(EEG_car, mode = 'within_ch')

        return EEG_epoch_norm, reject_mask
    
class Feature_Extraction():
    def __init__(self, fs = 250, window_size = 150, step_size = 75):
        self.fs = fs
        self.window_size = window_size
        self.step_size = step_size

    def window_epoch_equal_periods(self, data_epoch):
        """
        Only used for OpenBCI

        data_epoch: (1125, channels)
        returns:
            (num_windows_total, window_size, channels)
        """

        PERIOD = 375
        windows_all = []

        for i in range(3):  # rest, contract, release
            start = i * PERIOD
            end   = start + PERIOD

            segment = data_epoch[start:end]

            w = self.centered_sliding_windows(
                segment,
                self.window_size,
                self.step_size
            )

            windows_all.append(w)

        return np.concatenate(windows_all, axis=0)

    def centered_sliding_windows(self, data):
        """
        data: (samples, channels)

        Returns:
            windows: (n_windows, window_size, channels)
            centers: center sample indices
        """
        from numpy.lib.stride_tricks import sliding_window_view
        half = self.window_size // 2

        # symmetric padding
        padded = np.pad(                # Mirror the edges by half of the window_size. So array of [0 1 2 1 0] with window_size 3 becombes after padding [1 0 1 2 1 0 1]
            data,
            ((half, half), (0, 0)),
            mode='reflect'
        )
        
        windows = sliding_window_view(  # Slide over data with window size. The example from before with stepsize of 1: [1 0 1] -> [0 1 2] -> [1 2 1] -> [2 1 0] - > [1 0 1]
            padded,
            window_shape = self.window_size,
            axis=0
        )

        windows = windows[::self.step_size]


        return windows          # Shape (n_windows, window_size, channels)

    def extract_statistical_features(self, signal):
        mean = np.mean(signal)
        median = np.median(signal)
        std_dev = np.std(signal)
        variance = np.var(signal)
        skewness = stats.skew(signal)
        kurtosis = stats.kurtosis(signal)
        range_val = np.ptp(signal)
        abs_area = np.sum(np.abs(signal))
        
        features = {
            'Mean': mean,
            'Median': median,
            'Standard Deviation': std_dev,
            'Variance': variance,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Range': range_val,
            'Absolute Area' : abs_area
        }
        
        return features

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
        from scipy.signal import welch
        freqs, psd = welch(signal, self.fs)
        # dominant_freq = freqs[np.argmax(psd)]
        #total_power = np.sum(psd)
        band_power_delta = np.sum(psd[(freqs >= 0.5) & (freqs <= 4)])
        band_power_theta = np.sum(psd[(freqs >= 4) & (freqs <= 8)])
        band_power_alpha = np.sum(psd[(freqs >= 8) & (freqs <= 12)])
        band_power_beta = np.sum(psd[(freqs >= 12) & (freqs <= 30)])
        # mean_freq = np.mean(freqs)
        # median_freq = np.median(freqs)
        # peak_freq = freqs[np.argmax(psd)]
        # freq_variance = np.var(freqs)
        
        features = {
            'Band Power Delta' : band_power_delta,
            'Band Power Theta' : band_power_theta,
            'Band Power Alpha' : band_power_alpha, 
            'Band Power Beta' : band_power_beta
        }
        
        '''
        features = {
            'Dominant Frequency': dominant_freq,
            'Total Power': total_power,
            'Band Power (0.5-40 Hz)': band_power,
            'Mean Frequency': mean_freq,
            'Median Frequency': median_freq,
            'Peak Frequency': peak_freq,
            'Frequency Variance': freq_variance
        }
        '''
        return features

    def extract_entropy_features(self, signal):
        sample_entropy = ant.sample_entropy(signal)
        spectral_entropy = ant.spectral_entropy(signal, sf = self.fs, method='welch')
        perm_entropy = ant.perm_entropy(signal, normalize=True)
        svd_entropy = ant.svd_entropy(signal, order=3, delay=1)
        app_entropy = ant.app_entropy(signal)
        
        features = {
            'Sample Entropy': sample_entropy,
            'Spectral Entropy': spectral_entropy,
            'Permutation Entropy': perm_entropy,
            'SVD Entropy': svd_entropy,
            'Approximate Entropy': app_entropy
        }
        
        return features

    def extract_features_from_windows(self, X_epoch, feature_func_dict : Callable):

        num_epochs, num_windows, n_channels, _ = X_epoch.shape

        all_rows = []
        track_info = []

        for epo in range(num_epochs):
            for win in range(num_windows):

                row_features = {}
                row_info = {}

                for ch in range(n_channels):

                    signal = X_epoch[epo, win, ch, :]

                    for func_name, feature_func in feature_func_dict.items():
                        
                        feats = feature_func(signal)

                        # rename features per channel
                        for feat_name, value in feats.items():
                            col_name = f'ch{ch}_{func_name}_{feat_name}'
                            row_features[col_name] = value

                # optional tracking info
                row_info['epoch'] = epo
                row_info['window'] = win

                all_rows.append(row_features)
                track_info.append(row_info)

        df = pd.DataFrame(all_rows)
        df_info = pd.DataFrame(track_info)

        return df, df_info                  # df shape: (num_epochs * num_windows, num_features)

    def create_window_labels(self, df, windows_per_period):
        '''
        Only used for OpenBCI
        '''
        Y = np.zeros(len(df), dtype=int)
        Wpp = windows_per_period

        Y[df['window'] < Wpp] = 0          # Rest
        Y[(df['window'] >= Wpp) &
        (df['window'] < Wpp*2)] = 1        # Contract
        Y[df['window'] >= Wpp*2] = 2         # Release

        return Y

    def df_to_lstm(self, normalized_df, num_epochs, num_windows):

        feature_cols = [
            c for c in normalized_df.columns
        ]

        X = normalized_df[feature_cols].values

        X = X.reshape(num_epochs,
                    num_windows,
                    len(feature_cols))

        return X

    def feature_extraction_rutine(self, X_train_val_concat, X_test_concat, epoch_overview):
        # Concatinate classes with shape (epochs, samples, channels)
        EEG_slide_train_val = np.array([                        
            self.centered_sliding_windows(epo)
            for epo in X_train_val_concat
        ])  # Return (Epoch, windows, channel, window_size)

        EEG_slide_test = np.array([                        
            self.centered_sliding_windows(epo)
            for epo in X_test_concat
        ])  # Return (Epoch, windows, channel, window_size)

        T_epochs, T_windows, _, _ = EEG_slide_train_val.shape
        TE_epochs, TE_windows, _, _ = EEG_slide_test.shape

        feature_func_dict = {
        'stat' : self.extract_statistical_features,
        'time domain' : self.extract_time_domain_features,
        'freq domain' : self.extract_frequency_domain_features,           # NOTE: Set frequency in function
        'entropy' : self.extract_entropy_features                         # NOTE: Set frequency in function

    }

        EEG_train_val_feat, _ = self.extract_features_from_windows(X_epoch = EEG_slide_train_val,
                                                                    feature_func_dict = feature_func_dict
                                                                    ) # (num_epochs * num_windows, num_features)

        EEG_test_feat, _ = self.extract_features_from_windows(X_epoch = EEG_slide_test,
                                                                feature_func_dict = feature_func_dict
                                                                ) # (num_epochs * num_windows, num_features)

        #===========#
        # Normalize #
        #===========#
        # Initialize the StandardScaler
        scaler = StandardScaler()

        # Apply the scaler to normalize the data
        EEG_train_val_norm = scaler.fit_transform(EEG_train_val_feat)
        EEG_test_norm = scaler.fit_transform(EEG_test_feat)

        # Convert the result back into a DataFrame
        EEG_train_val_norm_df = pd.DataFrame(EEG_train_val_norm, columns = EEG_train_val_feat.columns)
        EEG_test_norm_df = pd.DataFrame(EEG_test_norm, columns = EEG_test_feat.columns)

        #===================#
        # Feature selection #
        #===================#
        LH_Train_epoch = epoch_overview['left_hand_train.csv']
        RH_Train_epoch = epoch_overview['right_hand_train.csv']
        F_Train_epoch = epoch_overview['feet_train.csv']
        T_Train_epoch = epoch_overview['tongue_train.csv']

        train_val_labels = np.concatenate([np.full(LH_Train_epoch*T_windows, 0),
                                           np.full(RH_Train_epoch*T_windows, 1),
                                           np.full(F_Train_epoch*T_windows, 2),
                                           np.full(T_Train_epoch*T_windows, 3)
                                           ])

        normalized_labeled_df = EEG_train_val_norm_df.copy()
        normalized_labeled_df['class'] = train_val_labels

        significant_features = []
        sig_count = 0
        for feature in EEG_train_val_norm_df.columns:
            class_1 = EEG_train_val_norm_df[feature][normalized_labeled_df['class'] == 0]
            class_2 = EEG_train_val_norm_df[feature][normalized_labeled_df['class'] == 1]
            class_3 = EEG_train_val_norm_df[feature][normalized_labeled_df['class'] == 2]
            class_4 = EEG_train_val_norm_df[feature][normalized_labeled_df['class'] == 3]
            
            stat, p = stats.kruskal(class_1, class_2, class_3, class_4)

            if p < 0.05:
                sig_count += 1
                significant_features.append(feature)
        print(f'Number of significant features {sig_count} / {EEG_test_feat.shape[1]}')

        # Extract those features which are significant - (num_epochs * num_windows, num_features)
        X_train = EEG_train_val_norm_df[significant_features]           
        X_test   = EEG_test_norm_df[significant_features]

        # Convert to (num epochs, windows, features)
        X_train = self.df_to_lstm(X_train, T_epochs, T_windows)
        X_test = self.df_to_lstm(X_test, TE_epochs, TE_windows)

        print('\n------Feature Extration------\n')
        print(f'Final X_train shape: {X_train.shape}')
        print(f'Final X_test shape: {X_test.shape}')

        return X_train, X_test

class DataAnalysis():
    def __init__(self, fs : int, base_dir : Path):
        self.fs = fs
        self.trim_period = 3
        self.trial_period = 9
        self.base_dir = base_dir
    
    def _preprocessing_routine(self, raw_eeg : np.ndarray, lowcut : int, highcut : int) -> tuple[np.ndarray, int]:
        '''
        Performs the full preprocessing routine:
        1) Notch + Bandpass filter
        2) Resample + z-score standardization + Secmentation into epochs

        Parameters
        ----------
        raw_eeg : np.ndarray
            This holds keys for a specfic class (finger). NOTE - If raw_eeg is a list, it will be converted to a dict with key 'single_class'. 2D array - Dim(samples, channels)

        Return
        ------
        :return: np.ndarray of normalized EEG data
        :return: Int of the total amount of epochs for one experiment
        '''
        # ---------------------------#
        # 1) NOTCH + BANDPASS FILTER #
        # ---------------------------#
        EEG_filter_ins = Filtering(fs = self.fs)
        
        EEG_notch = EEG_filter_ins.notch(data = raw_eeg, cutoff = 50, Q = 30)
        EEG_bandpass, _ = EEG_filter_ins.butter_bandpass(data = EEG_notch, lowcut = lowcut, highcut = highcut, order = 4)

        #===============================#
        # 2) Calculate number of epochs #
        #===============================#
        trim_samples = self.fs * self.trim_period           # 375
        samples_per_epoch = self.fs * self.trial_period

        valid_samples = EEG_bandpass.shape[0] - 2 * trim_samples            # Total samples for experimental period. WHY *2 : Trim egde on both sides
        num_epochs = int( np.round(valid_samples / samples_per_epoch) )     # Divide out total samples in sections of samples per epoch -> Results in number of epochs

        trim_start = trim_samples
        trim_end = trim_start + num_epochs * samples_per_epoch              # WHY instead of data[trim : -trim] -> Inconsistency in protocol causes the last batch of data not be included -> Rare but can happen

        if (trim_end - trim_start) % samples_per_epoch != 0:                # Inform if epochs is differnet from usual amount. Can happen if bad trials is removed.
            print(f"Warning: Samples not perfectly divisible by trial period. Calculated num epochs: {valid_samples / samples_per_epoch}")
            print(f'Trim samples at start and end: {trim_start}, {trim_end}\n')
            print(f"Total samples: {EEG_bandpass.shape[0]}, Valid samples: {valid_samples}, Samples per epoch: {samples_per_epoch}, Calculated num epochs: {num_epochs}")
        
        #=========#
        # 3) TRIM #
        #=========#       
        EEG_trim = EEG_bandpass[trim_start : trim_end, :]
        # print(f"Original shape {EEG_bandpass.shape}\n"
        #       f'EEG_trim shape: {EEG_trim.shape}\n')

        return EEG_trim, num_epochs
    
    def load_EEG_data(self, subject_name : str | list, finger_name : str, reject_config_dict : dict, lowcut : int, highcut : int):
        reject_ins = RejectBadEpochs(base_dir = self.base_dir)
        load_ins = load_datasets(base_dir = self.base_dir)

        #================#
        # Find EEG files #
        #================#
        EEG_files = load_ins.find_flex_files(
            subjects = subject_name,
            modality = "EEG",
            fingers = finger_name,
            prefix = 'flex'
        )

        eeg_data = []
        epochs_overview = []

        for data_file in EEG_files:
            print(data_file)
            raw_data_df = pd.read_csv(data_file)
            raw_data = raw_data_df.iloc[:, 1:17].to_numpy()
            
            # Preprocessing
            eeg_temp, num_epochs = self._preprocessing_routine(raw_eeg = raw_data, lowcut = lowcut, highcut = highcut)

            eeg_data.append(eeg_temp)
            epochs_overview.append(num_epochs)
            
        EEG = np.concatenate(eeg_data, axis = 0)

        # Should be in sherpa loop 
        reject_mask = reject_ins.reject_routine(data_file_per_finger = EEG_files,
                                                epochs_overview = epochs_overview,
                                                EEG_data = EEG,
                                                RMS_data = None,
                                                reject_config_dict = reject_config_dict,
                                                EEG_useable_channels = None)

        total_epochs = sum(epochs_overview)
        EEG_epoch = EEG.reshape(total_epochs, EEG.shape[0] // total_epochs, EEG.shape[1])

        EEG_epoch_clean = EEG_epoch[~reject_mask]

        EEG_car = EEG_epoch_clean - np.mean(EEG_epoch_clean, axis = 2, keepdims = True)

        filt_ins = Filtering()
        EEG_epoch_norm = filt_ins.zscore(EEG_car, mode = 'within_ch')

        return EEG_epoch_norm, epochs_overview

    def plot_bandpower_heatmaps(self, data, subjects, REGIONS, BANDS, class_names):
        """
        Plot heatmaps (Channels x Frequency bands) for each class.

        Parameters
        ----------
        data : list of tuples
            [(feature_dict, label), ...]
            feature_dict format:
                {channel: {band: value}}
        class_names : list
            Mapping {label: "name"}
        """

        # data shape : (subjects, classes, regions, bands)

        # -----------------------------
        # Group data by class
        # -----------------------------
        all_mats = []

        for subj_data in data:
            subj_mats = []

            for feat_dict in subj_data:                      # Extract PSD features (per channel x per band) for Class1 and then class2
                
                # Convert dict → matrix (channels × bands)
                mat = np.zeros((len(REGIONS), len(BANDS)))

                for i, ch in enumerate(REGIONS):
                    for j, band in enumerate(BANDS):
                        mat[i, j] = np.mean(feat_dict[ch][band])
                
                subj_mats.append(mat)
            
            all_mats.append(subj_mats)
        
        all_mats = np.array(all_mats)           # Shape: (Subj, class, region, band)
        
        S, C, R, B = all_mats.shape

        #======================================#
        # Normalize color scale across classes #
        #======================================#
        fig, axes = plt.subplots(nrows = C, ncols = S, figsize = (4*S, 3*C))

        # all_vals = []
        # for mats in class_data.values():
        #     all_vals.append(np.mean(np.array(mats), axis=0))

        vmin = np.min(all_mats)
        vmax = np.max(all_mats)

        # -----------------------------
        # Plot heatmap per class
        # -----------------------------
        for s in range(S):
            for c in range(C):
                ax = axes[c, s] if S > 1 else axes[c]

                mat = all_mats[s, c]            # (R, B)

                im = ax.imshow(mat, aspect='auto', vmin = vmin, vmax = vmax, cmap='viridis')

                for i in range(R):          # regions (rows)
                    for j in range(B):      # bands (cols)
                        val = mat[i, j]

                        ax.text(
                            j, i,
                            f"{val:.2f}",   # format (2 decimals)
                            ha='center',
                            va='center',
                            color='white' if val < (vmin + vmax)/2 else 'black',
                            fontsize=7
                        )
                
                # Titles (top row)
                if c == 0:
                    ax.set_title(subjects[s])

                # Y labels (left column)
                if s == 0:
                    ax.set_ylabel(class_names[c])

                # Axis ticks
                if c == C - 1:
                    ax.set_xticks(range(B))
                    ax.set_xticklabels(BANDS, rotation=45, ha='right', fontsize=8)
                else:
                    ax.set_xticks([])

                if s == 0:
                    ax.set_yticks(range(R))
                    ax.set_yticklabels(REGIONS, fontsize=9)
                else:
                    ax.set_yticks([])
                
        # One shared colorbar
        fig.subplots_adjust(right=0.88)  # make space on the right
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("PSD (µV²/Hz)", fontsize=10)

        # plt.tight_layout()
        # plt.savefig('band_power_heatmap_subjects_8-11.png', dpi = 400)
        plt.show()

    def segment_into_periods(self, epochs):
        rest = epochs[:, : 3*self.fs, :]
        contract = epochs[:, 3*self.fs : 6*self.fs, :]
        release =  epochs[:, 6*self.fs : , :]

        return rest, contract, release
    
    def compute_trialwise_region_psd(self, data_class : np.ndarray, REGIONS : Dict, FREQ_BANDS : Dict, EEG_channel_names : List, EEG_FREQ : int):
        """
        Compute trial-wise PSD features aggregated per region.

        Parameters
        ----------
        data_class : np.ndarray
            Shape (trials, samples, channels)

        Returns
        -------
        region_features : dict
            {region: {band: [values_per_trial]}}
        """

        region_features = {                         # Create a dict per region and its bands
        region: {band: [] for band in FREQ_BANDS.keys()}
        for region in REGIONS.keys()
        }

        N_trials = data_class.shape[0]

        for trial in range(N_trials):

            for region_name, ch_list in REGIONS.items():

                trial_band_values = {band: [] for band in FREQ_BANDS.keys()}    # Dict of freq bands

                for ch in ch_list:
                    ch_idx = EEG_channel_names.index(ch)                        # Extract index where channel belong

                    signal = data_class[trial, :, ch_idx]
                    
                    f, Pxx = welch(signal, fs = EEG_FREQ, nperseg = len(signal))

                    total_power = np.trapz(Pxx, f)

                    if total_power == 0:                                        # Avoid division by zero
                        continue

                    for freq_name, (low, high) in FREQ_BANDS.items():
                        idx = (f >= low) & (f <= high)

                        if np.any(idx):
                            band_power = np.trapz(Pxx[idx], f[idx])
                            rel_power = band_power / total_power
                            trial_band_values[freq_name].append(rel_power)      # Contain freq bands per channel, Like {'delta': [Fp1, Fp2], 'theta': [Fp1, Fp2], ...}
                
                for band in FREQ_BANDS.keys():
                    if len(trial_band_values[band]) > 0:

                        region_features[region_name][band].append(np.mean(trial_band_values[band]))
        
        return region_features
      
    def cohens_d(self, x1, x2):
        n1, s1, m1 = len(x1), np.std(x1), np.mean(x1)
        n2, s2, m2 = len(x2), np.std(x2), np.mean(x2)

        S_pool = np.sqrt( (s1**2 * (n1 - 1) + s2**2 * (n2 - 1)) / (n1 + n2 - 2) )
        
        if S_pool == 0:
            return 0
        
        return (m1 - m2) / S_pool
    
    def compute_multiclass_separability(self, data_classes, REGIONS, BANDS):
        class_pairs = [
            (0, 1),  # left_hand vs right_hand
            (0, 2),  # left_hand vs feet
            (0, 3),  # left_hand vs tongue
            (1, 2),  # right_hand vs feet
            (1, 3),  # right_hand vs tongue
            (2, 3)   # feet vs tongue
        ]
        comparison_names = [
        "left_hand vs right_hand",
        "left_hand vs feet",
        "left_hand vs tongue",
        "right_hand vs feet",
        "right_hand vs tongue",
        "feet vs tongue"
        ]
        
        d_all_subjects = []
        R = len(REGIONS)
        B = len(BANDS)

        for subj in data_classes:
            print(len(subj))

            d_subject = []

            for c1, c2 in class_pairs:
                d_map = np.zeros((R, B))

                for i, r in enumerate(REGIONS):
                    for j, b in enumerate(BANDS):
                        x1 = subj[c1][r][b]            # across trials (trials, r, b)
                        x2 = subj[c2][r][b]

                        d_map[i, j] = self.cohens_d(x1, x2)
                
                d_subject.append(d_map)
                
            d_all_subjects.append(d_subject)
        
        d_all_subjects = np.array(d_all_subjects)                       # shape: (subjects, comparisons, regions, bands)

        d_abs = np.abs(d_all_subjects)

        d_mean = np.mean(d_abs, axis=0)
        d_std  = np.std(d_abs, axis=0)

        return d_mean, d_std, comparison_names
    
def inspect_frequency_ranges():
    SAMPLES_PER_TRIAL = 1001
    EEG_FREQ = 250
    EEG_LOWCUT = 0.5
    EEG_HIGHCUT = 60
    REJECT_CONFIG_DICT = {
        'EEG_epoch_rejection_tolerance' : 6,
        'EMG_epoch_rejection_tolerance' : 6,
        'EEG_ch_acceptance' : 0,
        'EMG_ch_acceptance' : 0
    }
    train_or_test = ['train', 'test']

    base_dir = Path(__file__).resolve().parents[1] / 'utilities/BCI_IV_2a'         # Path(__file__).resolve() -> Absolute path to this src folder
    data_ins = DataAnalysis(fs = EEG_FREQ, base_dir = base_dir)

    # SUBJECT_NAME = ['subject_0', 'subject_1', 'subject_2', 'subject_3', 'subject_4', 'subject_5', 'subject_6', 'subject_7', 'subject_8', 'subject_9', 'subject_10', 'subject_11']
    FREQ_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 60)
    }
    REGIONS = {
    "frontal": ['Fz','FC3','FC1','FCz', 'FC2', 'FC4'],
    "central": ['C5','C3','C1','Cz','C2','C4','C6'],
    "parietal_central": ['CP3','CP1','CPz','CP2','CP4'],
    "parietal": ['P1','Pz','P2','POz']
    }
    EEG_channel_names = ['Fz','FC3','FC1','FCz', 'FC2', 'FC4', 'C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4', 'P1','Pz','P2','POz']

    select_channels = range(0, 22)

    BCI_preprocess_ins = BCI2a_preprocess(fs = EEG_FREQ,
                                        lowcut = EEG_LOWCUT,
                                        highcut = EEG_HIGHCUT,
                                        reject_config_dict = REJECT_CONFIG_DICT,
                                        samples_per_trial = SAMPLES_PER_TRIAL,
                                        EEG_useable_channels = None)

    SUBJECT_NAME = ['subject_1', 'subject_2', 'subject_3', 'subject_4', 'subject_5', 'subject_6', 'subject_7', 'subject_8', 'subject_9']     
    all_subject_data = []
    classes = ['left_hand', 'right_hand', 'feet', 'tongue']

    for subj in SUBJECT_NAME:
        
        # Initialize storage for this subject
        subject_dataset = {cls: [] for cls in classes}

        for tot in train_or_test:
            train_classes = [f'left_hand_{tot}', f'right_hand_{tot}', f'feet_{tot}', f'tongue_{tot}']

            for tc, dict_class in zip(train_classes, classes):
                print('\n-----------------------------------')
                print(f'Subject_{subj} - class : {tc}')
                
                dataset = base_dir / subj / (tc + '.csv')
                data = pd.read_csv(dataset).to_numpy()              # (samples, channels)

                EEG_temp = data[:, select_channels] 

                #===============#
                # Preprocessing #
                #===============#
                EEG_epoch_norm, _ = BCI_preprocess_ins.preprocess_BCI2a_dataset(EEG_temp)


                subject_dataset[dict_class].append(EEG_epoch_norm)
             

        #===========================#
        # Concatenate per class     #
        #===========================#
        for cls in subject_dataset:
            subject_dataset[cls] = np.concatenate(subject_dataset[cls], axis=0)

        data_classes = []                   # Container for classes with regions_features

        for data_class in subject_dataset.values():

            region_features = data_ins.compute_trialwise_region_psd(
                data_class = data_class,
                REGIONS = REGIONS,
                FREQ_BANDS = FREQ_BANDS,
                EEG_channel_names = EEG_channel_names,
                EEG_FREQ = EEG_FREQ
            )

            data_classes.append(region_features)                        

        # all_subjects_data =
        # [
        #     [dict_rest, dict_con, dict_rel],   # subject 0
        #     [dict_rest, dict_con, dict_rel],   # subject 1
        # ]
        all_subject_data.append(data_classes)

    all_subject_data = np.array(all_subject_data)
    
    data_ins.plot_bandpower_heatmaps(data = all_subject_data, subjects=SUBJECT_NAME, REGIONS = REGIONS, BANDS = FREQ_BANDS, class_names=classes)
    
    
    d_mean, d_std, comparison_names = data_ins.compute_multiclass_separability(all_subject_data, REGIONS=REGIONS, BANDS=FREQ_BANDS)
    
    
    all_mats = d_mean
    C, R, B = all_mats.shape
    S = 1
    #======================================#
    # Normalize color scale across classes #
    #======================================#
    fig, axes = plt.subplots(nrows = C, ncols = S, figsize = (8*S, 3*C))        # 4*S, 3*C

    vmin = np.min(all_mats)
    vmax = np.max(all_mats)

    # -----------------------------
    # Plot heatmap per class
    # -----------------------------
    for s in range(S):
        for c in range(C):
            ax = axes[c, s] if S > 1 else axes[c]

            mat = all_mats[c]            # (R, B)

            im = ax.imshow(mat, aspect='auto', vmin = vmin, vmax = vmax, cmap='viridis')

            for i in range(R):          # regions (rows)
                for j in range(B):      # bands (cols)
                    val = mat[i, j]

                    ax.text(
                        j, i,
                        f"{val:.2f}",   # format (2 decimals)
                        ha='center',
                        va='center',
                        color='white' if val < (vmin + vmax)/2 else 'black',
                        fontsize=7
                    )
            
            # Titles (top row)
            
            ax.set_title(comparison_names[c])

            # Y labels (left column)
            if s == 0:
                ax.set_ylabel('Regions')

            # Axis ticks
            if c == C - 1:
                ax.set_xticks(range(B))
                ax.set_xticklabels(FREQ_BANDS, rotation=45, ha='right', fontsize=8)
            else:
                ax.set_xticks([])

            if s == 0:
                ax.set_yticks(range(R))
                ax.set_yticklabels(REGIONS, fontsize=9)
            else:
                ax.set_yticks([])
            
    # One shared colorbar
    fig.subplots_adjust(right=0.88)  # make space on the right
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Cohen's d", fontsize=10)

    # plt.tight_layout()
    # plt.savefig('cohen_d_across_subject_.png', dpi = 400)
    plt.show()
    # 0.2 = Small effect
    # 0.5 = Moderate effect
    # 0.8 = Large effect 
    # for region, bands in d_mean.items():
    #     print(f"\nRegion: {region}")

    #     for band, d in bands.items():
    #         print(f"  {band}: {d:.3f}")

def load_BCI2a_dataset(subject_id):
    data_dir = Path(__file__).resolve().parents[1] / f'utilities/BCI_IV_2a/{subject_id}'         # Path(__file__).resolve() -> Absolute path to this src folder

    train_or_test = ['train', 'test']
    SAMPLES_PER_TRIAL = 1001
    EEG_FREQ = 250
    EEG_LOWCUT = 8
    EEG_HIGHCUT = 13
    REJECT_CONFIG_DICT = {
        'EEG_epoch_rejection_tolerance' : 6,
        'EEG_ch_acceptance' : 0
    }
    raw_data_dict = {}
    data_dict = {}
    reject_mask_dict = {}
    epoch_overview = {}
    
    BCI_preprocess_ins = BCI2a_preprocess(fs = EEG_FREQ,
                                        lowcut = EEG_LOWCUT,
                                        highcut = EEG_HIGHCUT,
                                        reject_config_dict = REJECT_CONFIG_DICT,
                                        samples_per_trial = SAMPLES_PER_TRIAL,
                                        EEG_useable_channels = None)
    # Fz,FC3,FC1,FCz,FC2,FC4,C5,C3,C1,Cz,C2,C4,C6,CP3,CP1,CPz,CP2,CP4,P1,Pz,P2,POz,EOG1,EOG2,EOG3,stim
    select_channels = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    for tot in train_or_test:
        train_classes = [f'left_hand_{tot}.csv', f'right_hand_{tot}.csv', f'feet_{tot}.csv', f'tongue_{tot}.csv']

        for tc in train_classes:
            print('\n-----------------------------------')
            print(f'Subject_{subject_id} - class : {tc}')
            dataset = data_dir / tc
            data = pd.read_csv(dataset).to_numpy()              # (samples, channels)

            EEG_temp = data[:, select_channels]                            # Remove non EEG data

            #===============#
            # Preprocessing #
            #===============#
            EEG_epoch_norm, reject_mask = BCI_preprocess_ins.preprocess_BCI2a_dataset(EEG_temp)
            
            # Save into dict
            raw_data_dict[tc] = EEG_temp
            data_dict[tc] = EEG_epoch_norm
            reject_mask_dict[tc] = reject_mask
            epoch_overview[tc] = EEG_epoch_norm.shape[0]
    
    return data_dict, reject_mask_dict, raw_data_dict, epoch_overview

def load_classfication(subject_name : str | list, sherpa_log_folder : str = 'SingleNet_LSTM_EMG', model_name : str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()              # Use pin_memory if CUDA is available
    print(f"Using device: {device}")
    print("Pin memory set to:", pin_memory)

    LOG_NAME = f'{subject_name}'
    log_dir = Path(__file__).resolve().parent / f'loggings/BCI_IV_2a/{sherpa_log_folder}/{LOG_NAME}'         # Path(__file__).resolve() -> Absolute path to this file

    #==========================#
    # NOTE: Tensorboard config #
    #==========================#
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')                                        # Use when having tensorboard
    os.makedirs(log_dir, exist_ok=False)    

    logger = ExperimentLogger(save_path = log_dir)
    split_ins = Manage2Split(seed = SEED)
    train_eval_ins = SingleNet_train_eval()
    model_handler_ins = SingleNetHandler(model_name = model_name, sensor_name = 'EEG')

    #===========#
    # Load data #
    #===========#
    dataset, _, _, epochs_overview = load_BCI2a_dataset(subject_id = subject_name)

    LH_Train = dataset['left_hand_train.csv']
    RH_Train = dataset['right_hand_train.csv']
    F_Train = dataset['feet_train.csv']
    T_Train = dataset['tongue_train.csv']

    LH_Test = dataset['left_hand_test.csv']
    RH_Test = dataset['right_hand_test.csv']
    F_Test = dataset['feet_test.csv']
    T_Test = dataset['tongue_test.csv']

    # PLOT DATA
    # LH_raw = raw['left_hand_train.csv']
    # reject_mask_indices = np.where(LH_reject)[0]
    # LH_reject = reject['left_hand_train.csv']
    # vis_ins = visualize_EEG(fs = 250, trial_period = 4, BCI2a_or_OpenBCI='BCI2a')
    # vis_ins.plot_egg_across_channels(LH_raw, 0, 0, None, 3, bad_epochs = reject_mask_indices)
    
    # Concat of class 1 and class 2
    X_train_val_concat, y_train_val_concat = split_ins.build_split(epoch_class1 = LH_Train, epoch_class2 = RH_Train, epoch_class3 = F_Train, epoch_class4 = T_Train)
    X_test_concat, y_test_concat = split_ins.build_split(epoch_class1 = LH_Test, epoch_class2 = RH_Test, epoch_class3 = F_Test, epoch_class4 = T_Test)

    # Indicies for split of dataset
    train_rng_indices, val_rng_indices = split_ins.split_trials(num_epochs = X_train_val_concat.shape[0])
    test_rng_indices = split_ins.shuffle_BCI2a_test(num_epochs = X_test_concat.shape[0])
    
    # Create whole dataset
    X_train_val_ins = SingleManageDataset(X_train_val_concat, y_train_val_concat, data_type='BCI_IV_2a')
    X_test_ins = SingleManageDataset(X_test_concat, y_test_concat, data_type='BCI_IV_2a')

    # Shuffle dataset with random indices
    X_train = Subset(X_train_val_ins, train_rng_indices)
    X_val = Subset(X_train_val_ins, val_rng_indices)
    X_test = Subset(X_test_ins, test_rng_indices)

    #========================================================#
    # THESE PARAMETERS ARE CHANCEABLE, DEPENDING ON THE TASK #
    #========================================================#
    MAX_NUM_TRIALS = 100             # 75 - 250 (simply to max)      # EMG 100 - EEG 250
    NUM_INITIAL_DATA_POINTS = 75
    DATA_CH = X_train_val_concat.shape[2]
    NUM_CLASSES = 4
    NUM_EPOCHS = 250                 # 150 - 200                    # EMG 150 - EEG 200
    PATIENCE = 25                   # Early stopping patience - 25

    #===========#
    # Constants #
    #===========#
    global_best_vloss = float("inf")                # Used to only save one model.

    #====================================#
    # SHERPA Hyperparameter Optimazation #
    #====================================#

    parameters = model_handler_ins.get_hyperparameters()
    
    # algorithm = sherpa.algorithms.RandomSearch(max_num_trials = MAX_NUM_TRIALS)
    algorithm = sherpa.algorithms.GPyOpt(
        max_num_trials = MAX_NUM_TRIALS,
        acquisition_type = 'EI',                     # Expected improvement
        num_initial_data_points = NUM_INITIAL_DATA_POINTS                 # Number of hyperparameter configurations before model learns
    )
    # Study represents the hyperparameter optimization itself
    study = sherpa.Study(
        parameters = parameters,
        algorithm = algorithm,
        lower_is_better = True,
        disable_dashboard = True
    )

    for trial in study:
        model_config = model_handler_ins.build_model_config(
            trial = trial,
            input_dim = DATA_CH,
            TOTAL_CLASSES = NUM_CLASSES
        )
        train_config = model_handler_ins.build_training_config(
            trial = trial
        )

        #=================#
        # Single datasets #
        #=================#
        model = model_handler_ins.get_model(config = model_config)
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params = model.parameters(), lr = train_config['lr'], weight_decay = train_config['weight_decay'])

        # Create data loader
        train_loader = DataLoader(X_train, batch_size = train_config['batch_size'], shuffle = True, pin_memory = pin_memory, num_workers = 0)
        val_loader = DataLoader(X_val, batch_size = train_config['batch_size'], shuffle = False, pin_memory = pin_memory, num_workers = 0)
        test_loader = DataLoader(X_test, batch_size = train_config['batch_size'], shuffle = False, pin_memory = pin_memory, num_workers = 0)

        best_train_loss = None
        best_val_loss = float("inf")
        best_val_acc = None

        best_epoch = 0
        best_state_dict = None
        early_stopping_counter = 0

        #===================================#
        # NOTE: Tensorboard config          #
        #   Enable all if using tensorboard #
        #===================================#
        # log_folder = os.path.join(log_dir, f"trial_{trial.id}")               
        # os.makedirs(log_folder, exist_ok=False)
        # writer = SummaryWriter(os.path.join(log_folder, 'trial_{}_timestamp_{}'.format(trial.id, timestamp)))

        for epoch in range(NUM_EPOCHS):

            # Train model
            avg_train_loss = train_eval_ins.train_one_epoch(model = model, train_loader = train_loader, criterion = criterion, optimizer = optimizer, device = device)

            # Validate model
            avg_vloss, vacc, _ = train_eval_ins.validaton_one_epoch(model = model, val_loader = val_loader, criterion = criterion, device = device)

            # Tensor Board logging
            # writer.add_scalars('Loss', { 'Training' : avg_train_loss, 'Validation' : avg_vloss }, epoch + 1)
            # writer.add_scalars('Accuracy Validation', {'Validation' : vacc }, epoch + 1)
            # writer.flush()

            study.add_observation(trial = trial,
                                iteration = epoch,
                                objective = avg_vloss)

            # Track best performance, and save the model's state
            if avg_vloss < best_val_loss:
                best_val_loss = avg_vloss
                best_epoch = epoch

                best_state_dict = deepcopy(model.state_dict())
                best_optimizer_dict = deepcopy(optimizer.state_dict())
 
                best_train_loss = avg_train_loss
                best_val_acc = vacc

                early_stopping_counter = 0

            else:
                early_stopping_counter += 1

                if early_stopping_counter >= PATIENCE:
                    break

            print(
                f'{subject_name} | '
                f'Trial {trial.id}/{MAX_NUM_TRIALS} | '
                f'Epoch {epoch+1}/{NUM_EPOCHS} | '
                f'Train {avg_train_loss:.4f} | '
                f'Val {avg_vloss:.4f} | '
                f'Acc {vacc:.2f} |',
                f'Early stopping {early_stopping_counter} |',
                end='\r',
                flush=True
            )

        model.load_state_dict(best_state_dict)

        avg_test_loss, test_acc, predictions, labels = train_eval_ins.inference_one_epoch(model = model, test_loader = test_loader, criterion = criterion, device = device)
        
        logger.log_trial(
            trial_id=trial.id,
            hyperparams = trial.parameters,
            best_epoch = best_epoch,
            train_loss = best_train_loss,
            val_loss = best_val_loss,
            val_acc = best_val_acc,
            test_loss = avg_test_loss,
            test_acc = test_acc,
            preds = predictions,
            labels = labels)
        
        if best_val_loss < global_best_vloss:
            global_best_vloss = best_val_loss

            torch.save({
                "model_name": model_name,
                "sensor_name": 'EEG',
                "model_state": best_state_dict,
                "model_args": model_config, 
                "optimizer_state_dict": best_optimizer_dict,
                "hyperparameters": trial.parameters,}, 
                r'{}\model.pth'.format(log_dir))
        
        #writer.close()
        study.finalize(trial, status = 'COMPLETED')

def main():
    subject_id = ['subject_1', 'subject_2']            # subject_1 to subject_9

    sherpa_log_folder = 'SingleNet_LSTM'
    model_name = 'SingleNet_LSTM'

    for subj in subject_id:
        load_classfication(subject_name = subj, sherpa_log_folder = sherpa_log_folder, model_name = model_name)
    

if __name__ == '__main__':
    main()
    # inspect_model(logging_name = 'BCI_IV_2a/subject_3_CNN')
    # inspect_frequency_ranges()
