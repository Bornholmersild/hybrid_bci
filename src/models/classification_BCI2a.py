# Classification
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import sherpa

# Manage datasets
import numpy as np
import pandas as pd

# Manage utils
import os 
from pathlib import Path
from datetime import datetime
import logging                  # Avoid loggings from GP
from copy import deepcopy       # Used for copy model_state
import time

# Feature extraction
import scipy.stats as stats
from typing import Callable
from sklearn.preprocessing import StandardScaler
import antropy as ant

# Own implementations
from src.utilities.preprocessing import EEG_preprocessing, EMG_preprocessing, RejectBadEpochs, Filtering #E402
from src.utilities.trainer_and_evaluator import SingleNet_train_eval
from src.utilities.load_and_visualize_data import load_datasets, visualize_EEG
from src.models.classification_pipeline import SingleNet, SingleNet_CNN, SingleManageDataset, ExperimentLogger, SingleManageDataset, build_optimizer, inspect_model

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

        # EEG_epoch_norm = EEG_filter_ins.zscore(EEG_car, mode = 'within_ch')

        return EEG_car, reject_mask
    
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
    
def load_BCI2a_dataset(subject_id):
    data_dir = Path(__file__).resolve().parents[1] / f'utilities/BCI_IV_2a/{subject_id}'         # Path(__file__).resolve() -> Absolute path to this src folder

    train_or_test = ['train', 'test']
    SAMPLES_PER_TRIAL = 1001
    EEG_FREQ = 250
    EEG_LOWCUT = 6
    EEG_HIGHCUT = 32
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
    select_channels = [7, 9, 11]
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

def load_classfication(subject_id : str | list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()              # Use pin_memory if CUDA is available
    print(f"Using device: {device}")
    print("Pin memory set to:", pin_memory)

    LOG_NAME = f'{subject_id}_CNN'
    log_dir = Path(__file__).resolve().parent / f'loggings/BCI_IV_2a/{LOG_NAME}'         # Path(__file__).resolve() -> Absolute path to this file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    window_size = 150           # 150 - Openbci
    step_size = 75              # 75 - Openbci

    logger = ExperimentLogger(save_path = log_dir)
    split_ins = Manage2Split(seed = SEED)
    train_eval_ins = SingleNet_train_eval()
    # feat_ins = Feature_Extraction(fs = 250, window_size = window_size, step_size = step_size)

    #===========#
    # Load data #
    #===========#
    dataset, _, _, epochs_overview = load_BCI2a_dataset(subject_id = subject_id)

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

    #====================#
    # Feature Extraction #
    #====================#
    # X_train_val_concat, X_test_concat = feat_ins.feature_extraction_rutine(X_train_val_concat = X_train_val_concat,
    #                                                                        X_test_concat = X_test_concat,
    #                                                                        epoch_overview = epochs_overview)

    # Indicies for split of dataset
    train_rng_indices, val_rng_indices = split_ins.split_trials(num_epochs = X_train_val_concat.shape[0])
    test_rng_indices = split_ins.shuffle_BCI2a_test(num_epochs = X_test_concat.shape[0])
    
    # Create whole dataset
    X_train_val_ins = SingleManageDataset(X_train_val_concat, y_train_val_concat)
    X_test_ins = SingleManageDataset(X_test_concat, y_test_concat)

    # Shuffle dataset with random indices
    X_train = Subset(X_train_val_ins, train_rng_indices)
    X_val = Subset(X_train_val_ins, val_rng_indices)
    X_test = Subset(X_test_ins, test_rng_indices)

    #========================================================#
    # THESE PARAMETERS ARE CHANCEABLE, DEPENDING ON THE TASK #
    #========================================================#
    MAX_NUM_TRIALS = 100             # 75 - 250 (simply to max)      # EMG 100 - EEG 250
    DATA_CH = X_train_val_concat.shape[2]
    NUM_CLASSES = 2
    NUM_EPOCHS = 250                 # 150 - 200                    # EMG 150 - EEG 200
    PATIENCE = 40                   # Early stopping patience - 25
    
    #====================================#
    # SHERPA Hyperparameter Optimazation #
    #====================================#

    parameters = [sherpa.Continuous(name='learning_rate', range=[0.00001, 0.001], scale='log'),
              sherpa.Continuous(name='dropout', range=[0.1, 0.5]),
              sherpa.Ordinal(name='batch_size', range=[16, 32, 64]),
              sherpa.Discrete(name='num_hidden_units', range=[32, 256]),         # before 256 (EMG -> 64)
              sherpa.Choice(name='activation', range=['relu', 'elu']),
              sherpa.Ordinal(name='lstm_layers', range=[1, 3]),
              sherpa.Choice(name="optimizer", range=["adamw", "sgd_momentum"]),
              sherpa.Continuous(name="weight_decay", range=[1e-6, 1e-2], scale="log"),
              sherpa.Continuous(name="momentum", range=[0.7, 0.99]),   # only used for SGD
              sherpa.Choice(name="nesterov", range=[False, True]),     # only used for SGD])
              sherpa.Choice(name='cnn_filters', range=[16, 32, 64]),  # Only used for CNN
              sherpa.Choice(name='kernel_size', range=[25, 50, 75]),  # Only used for CNN
    ]
    
    # algorithm = sherpa.algorithms.RandomSearch(max_num_trials = MAX_NUM_TRIALS)
    algorithm = sherpa.algorithms.GPyOpt(
        max_num_trials = MAX_NUM_TRIALS,
        acquisition_type = 'EI',                     # Expected improvement
        num_initial_data_points = 10                 # Number of hyperparameter configurations before model learns
    )
    # Study represents the hyperparameter optimization itself
    study = sherpa.Study(
        parameters = parameters,
        algorithm = algorithm,
        lower_is_better = True,
        disable_dashboard = True
    )

    for trial in study:
        dropout = trial.parameters['dropout']       
        batch_size = trial.parameters['batch_size']
        num_hidden_units = trial.parameters['num_hidden_units']
        activation = trial.parameters['activation'] 
        lstm_layers = trial.parameters['lstm_layers']     # trial.parameters['lstm_layers']
        cnn_filters = trial.parameters['cnn_filters']
        kernel_size = trial.parameters['kernel_size']

        #=================#
        # Single datasets #
        #=================#
        model = SingleNet_CNN(data_ch = DATA_CH, num_classes = NUM_CLASSES, dropout = dropout, activation = activation, cnn_filters = cnn_filters, kernel_size = kernel_size)
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(model_params = model.parameters(), trial_parameters = trial.parameters)

        # Create data loader
        train_loader = DataLoader(X_train, batch_size = batch_size, shuffle = True, pin_memory = pin_memory, num_workers = 0)
        val_loader = DataLoader(X_val, batch_size = batch_size, shuffle = False, pin_memory = pin_memory, num_workers = 0)
        test_loader = DataLoader(X_test, batch_size = batch_size, shuffle = False, pin_memory = pin_memory, num_workers = 0)

        best_train_loss = None
        best_val_loss = float("inf")
        best_val_acc = None

        best_epoch = 0
        best_state_dict = None
        early_stopping_counter = 0

        # Create log folder
        log_folder = os.path.join(log_dir, f"trial_{trial.id}")
        os.makedirs(log_folder, exist_ok=False)
        writer = SummaryWriter(os.path.join(log_folder, 'trial_{}_timestamp_{}'.format(trial.id, timestamp)))

        for epoch in range(NUM_EPOCHS):

            # Train model
            avg_train_loss = train_eval_ins.train_one_epoch(model = model, train_loader = train_loader, criterion = criterion, optimizer = optimizer, device = device)

            # Validate model
            avg_vloss, vacc, _ = train_eval_ins.validaton_one_epoch(model = model, val_loader = val_loader, criterion = criterion, device = device)

            # Tensor Board logging
            writer.add_scalars('Loss', { 'Training' : avg_train_loss, 'Validation' : avg_vloss }, epoch + 1)
            writer.add_scalars('Accuracy Validation', {'Validation' : vacc }, epoch + 1)
            writer.flush()

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
                f'{subject_id} | '
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
        
        model_arg = {
            "data_ch": DATA_CH,
            "hidden": num_hidden_units,
            "lstm_layers": lstm_layers,
            "num_classes": NUM_CLASSES,
            "dropout": dropout,
            "activation": activation,}
        
            
        torch.save({
            "model_state": best_state_dict,
            "model_args": model_arg, 
            "optimizer_state_dict": best_optimizer_dict,
            "hyperparameters": trial.parameters,}, 
            r'{}\model.pth'.format(log_folder))
        
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

        writer.close()
        study.finalize(trial, status = 'COMPLETED')

def main():
    subject_id = 'subject_3'            # subject_1 to subject_9

    load_classfication(subject_id = subject_id)
    

if __name__ == '__main__':
    # main()
    inspect_model(logging_name = 'BCI_IV_2a/subject_3_CNN')
