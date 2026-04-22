# Manage directory
from pathlib import Path
from typing import List, Union
from collections.abc import Callable

# Manage plots
from flask import json
import matplotlib.pyplot as plt

# Manage data
import pandas as pd
import numpy as np

from scipy.signal import welch
# From own implementations
from src.utilities.preprocessing import Filtering, EEG_preprocessing, EMG_preprocessing, RejectBadEpochs
# from src.utilities.EEG_feature_extraction import FeatureExtraction

from typing import Dict, List

EEG_channel_names = [
    "Fp1", "Fp2",   # frontal pole
    "C3",  "C4",    # central
    "T5",  "T6",    # temporal (posterior)
    "Cz",  "Fz",    # occipital
    "F7",  "F8",    # temporal (anterior)
    "F3",  "F4",    # frontal
    "T3",  "T4",    # temporal (mid)
    "P3",  "P4"     # parietal
    ]

class load_datasets():
    '''
    Class to find data files and load EMG and EEG data

    Parameters
    ----------
    base_dir : Path
        Root directory of datasets (e.g. Path('experiment/data'))
    '''
    def __init__(self, base_dir : Path):
        self.base_dir = base_dir

    def find_flex_files(self,
                        subjects: Union[str, List[str]],
                        modality: str,
                        fingers: Union[str, List[str]],
                        prefix: str = "flex"
                        ) -> List[Path]:
        """
        Find flex CSV files for selected subjects, modality, and fingers.
        Reuse return files to load EEG or EMG, using load_datasets_EEG or load_datasets_EMG

        Parameters
        ----------
        subjects : list[str]
            List like ['subject_0', 'subject_1']
        modality : str
            'EEG' or 'EMG'
        fingers : str or list[str]
            'index', 'thumb', or ['index', 'thumb']
        prefix : str
            Filename prefix (default='flex')

        Returns
        -------
        list[Path]
            List of matching CSV file paths
        """

        if isinstance(fingers, str):
            fingers = [fingers]
        if isinstance(subjects, str):
            subjects = [subjects]

        paths = []

        for subject in subjects:
            data_dir = self.base_dir / subject / modality
            if not data_dir.exists():
                raise FileNotFoundError(f'{data_dir} - does not exists')

            for finger in fingers:
                pattern = f"{prefix}_{finger}_finger*.csv"
                paths.extend(data_dir.glob(pattern))

        return sorted(paths)
    
    def load_datasets_marker(self, path_to_data_files : Union[list | Path]):

        marker_dict = {}
        file_idx = 0

        for data_file in path_to_data_files:
             marker_dict[file_idx] = pd.read_csv(data_file)
             file_idx += 1

        return marker_dict
    '''
    def load_datasets_EEG(self,
                          path_to_data_files : Union[list | Path],
                          preprocessing_func : Callable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        # Load EEG data set given desired data path files from -> find_flex_files

        # Parameters
        # ----------
        # path_to_data_files : list
        #     A list of pathways to data files
        # bandpass_lowcut : int
        #     Desired lowcut bandpass frequency
        # bandpass_highcut : int
        #     Desired highcut bandpass frequency
        # extract_event : str
        #     Extract period of events (For example: 'all', 'contract', 'release', 'rest') by default 'all'

        # Returns
        # -------
        # :return: Continues preprocessed EEG data
        # :return: Epoch preprocessed EEG data
        # :return: Mean epoch preprocessed EEG data
        
        if isinstance(path_to_data_files, Path):
            path_to_data_files = [path_to_data_files]
        print('------------------\n'
              'Process for EEG data\n'
              '------------------\n')

        epochs_overview = []
        all_data = []

        for data_file in path_to_data_files:
            print(data_file)
        
            EEG_df = pd.read_csv(data_file)
            EEG_raw = EEG_df.iloc[:, 1:17].to_numpy()
            
            # Preprocessing
            EEG_filt, num_epochs = preprocessing_func(raw_eeg = EEG_raw)

            all_data.append(EEG_filt)
            epochs_overview.append(num_epochs)

        EEG = np.concatenate(all_data, axis = 0)
        print(f"Reshaped data shape: {EEG.shape}")

        return EEG, epochs_overview'''
    
    def load_EEG_data(self, subject_name : str | list, finger_name : str, reject_config_dict : dict, preprocessing_func : Callable, EEG_useable_channels : list | None):
        reject_ins = RejectBadEpochs(base_dir = self.base_dir)

        #================#
        # Find EEG files #
        #================#
        EEG_files = self.find_flex_files(
            subjects = subject_name,
            modality = "EEG",
            fingers = finger_name,
            prefix = 'flex'
        )

        EEG, epochs_overview = self._extract_EEG_data(path_to_data_files = EEG_files, 
                                                      preprocessing_func = preprocessing_func)

        # Should be in sherpa loop 
        reject_mask = reject_ins.reject_routine(data_file_per_finger = EEG_files,
                                                epochs_overview = epochs_overview,
                                                EEG_data = EEG,
                                                RMS_data = None,
                                                reject_config_dict = reject_config_dict,
                                                EEG_useable_channels = EEG_useable_channels)
        
        EEG = EEG[:, EEG_useable_channels].copy() if EEG_useable_channels is not None else EEG.copy()

        total_epochs = sum(epochs_overview)
        EEG_epoch = EEG.reshape(total_epochs, EEG.shape[0] // total_epochs, EEG.shape[1])

        EEG_epoch_clean = EEG_epoch[~reject_mask]

        EEG_car = EEG_epoch_clean - np.mean(EEG_epoch_clean, axis = 2, keepdims = True)

        filt_ins = Filtering()
        EEG_epoch_norm = filt_ins.zscore(EEG_car, mode = 'within_ch')

        return EEG_epoch_norm, epochs_overview
    
    def _extract_EEG_data(self,
                          path_to_data_files : Union[list | Path],
                          preprocessing_func : Callable) -> tuple[np.ndarray, np.ndarray, list]:
        print('------------------\n'
            'Process for EEG data\n'
            '------------------\n')
        if isinstance(path_to_data_files, Path):
            path_to_data_files = [path_to_data_files]

        epochs_overview = []
        eeg_data = []

        for data_file in path_to_data_files:
            print(data_file)
            raw_data_df = pd.read_csv(data_file)
            raw_data = raw_data_df.iloc[:, 1:17].to_numpy()
            
            # Preprocessing
            eeg_temp, num_epochs = preprocessing_func(raw_eeg = raw_data)

            eeg_data.append(eeg_temp)
            epochs_overview.append(num_epochs)
        
        EEG = np.concatenate(eeg_data, axis = 0)
        
        return EEG, epochs_overview
        
    def load_EMG_data(self, subject_name : str | list, finger_name : str, EMG_config_dict : dict, reject_config_dict : dict, preprocessing_func : Callable):
    
        reject_ins = RejectBadEpochs(base_dir = self.base_dir)

        #================#
        # Find EMG files #
        #================#
        EMG_files = self.find_flex_files(
            subjects = subject_name,
            modality = "EMG",
            fingers = finger_name,
            prefix = 'flex'
        )

        RMS, EMG, epochs_overview = self._extract_EMG_data(path_to_data_files = EMG_files, 
                                                           preprocessing_func = preprocessing_func,
                                                           EMG_config_dict = EMG_config_dict)

        # Should be in sherpa loop 
        reject_mask = reject_ins.reject_routine(data_file_per_finger = EMG_files,
                                                epochs_overview = epochs_overview,
                                                EEG_data = None,
                                                RMS_data = RMS,
                                                reject_config_dict = reject_config_dict,
                                                EEG_useable_channels = None)

        total_epochs = sum(epochs_overview)
        RMS_epoch = RMS.reshape(total_epochs, RMS.shape[0] // total_epochs, RMS.shape[1])
        EMG_epoch = EMG.reshape(total_epochs, EMG.shape[0] // total_epochs, EMG.shape[1]) if EMG is not None else None

        RMS_epoch_clean = RMS_epoch[~reject_mask]
        EMG_epoch_clean = EMG_epoch[~reject_mask] if EMG is not None else None

        filt_ins = Filtering()
        RMS_epoch_norm = filt_ins.zscore(RMS_epoch_clean, mode = 'within_ch')
        EMG_epoch_norm = filt_ins.zscore(EMG_epoch_clean, mode = 'within_ch') if EMG is not None else None

        return RMS_epoch_norm, EMG_epoch_norm, epochs_overview
    
    def _extract_EMG_data(self,
                          path_to_data_files : Union[list | Path],
                          preprocessing_func : Callable,
                          EMG_config_dict : dict) -> tuple[np.ndarray, np.ndarray, list]:
        print('------------------\n'
            'Process for EMG data\n'
            '------------------\n')
        if isinstance(path_to_data_files, Path):
            path_to_data_files = [path_to_data_files]

        epochs_overview = []
        rms_data = []
        emg_data = []

        for data_file in path_to_data_files:
            print(data_file)
            raw_data = pd.read_csv(data_file).to_numpy()
            
            # Preprocessing
            rms_temp, emg_temp, num_epochs = preprocessing_func(
                raw_emg = raw_data,
                rms_windowsize = EMG_config_dict['rms_windowsize'],
                rms_stepsize = EMG_config_dict['rms_stepsize'],
                hampel_windowsize = EMG_config_dict['hampel_windowsize'],
                hampel_sigma = EMG_config_dict['hampel_sigma'],
                hampel_plot_option = EMG_config_dict['hampel_plot_option']
            )

            rms_data.append(rms_temp)
            emg_data.append(emg_temp)
            epochs_overview.append(num_epochs)
        
        EMG = np.concatenate(emg_data, axis = 0) if EMG_config_dict['include_EMG'] else None
        RMS = np.concatenate(rms_data, axis = 0)
        
        return RMS, EMG, epochs_overview

    def load_EEG_EMG_data(self, 
                          subject_name : str | list,
                          finger_name : str,
                          reject_config_dict : dict,
                          EEG_preprocessing_func : Callable,
                          EMG_preprocessing_func : Callable,
                          EMG_config_dict : dict,
                          EEG_useable_channels : list | None) -> tuple[list, list, list, int]:

        reject_ins = RejectBadEpochs(base_dir = self.base_dir)

        #=================#
        # Find data files #
        #=================#
        EEG_files = self.find_flex_files(
            subjects = subject_name,
            modality = "EEG",
            fingers = finger_name,
            prefix = 'flex'
        )

        EMG_files = self.find_flex_files(
            subjects = subject_name,
            modality = "EMG",
            fingers = finger_name,
            prefix = 'flex'
        )        

        EEG, RMS, EMG, epochs_overview = self._extract_EEG_EMG_data(path_to_EEG_files = EEG_files,
                                                                    path_to_EMG_files = EMG_files,
                                                                    EEG_preprocessing_func = EEG_preprocessing_func,
                                                                    EMG_preprocessing_func = EMG_preprocessing_func,
                                                                    EMG_config_dict = EMG_config_dict)
        
        reject_mask = reject_ins.reject_routine(data_file_per_finger = EEG_files,
                                                epochs_overview = epochs_overview,
                                                EEG_data = EEG,
                                                RMS_data = RMS,
                                                reject_config_dict = reject_config_dict,
                                                EEG_useable_channels = EEG_useable_channels)
    
    
        EEG = EEG[:, EEG_useable_channels].copy() if EEG_useable_channels is not None else EEG.copy()

        total_epochs = sum(epochs_overview)
        EEG_epoch = EEG.reshape(total_epochs, EEG.shape[0] // total_epochs, EEG.shape[1])
        RMS_epoch = RMS.reshape(total_epochs, RMS.shape[0] // total_epochs, RMS.shape[1])
        EMG_epoch = EMG.reshape(total_epochs, EMG.shape[0] // total_epochs, EMG.shape[1]) if EMG is not None else None

        EEG_epoch_clean = EEG_epoch[~reject_mask]
        RMS_epoch_clean = RMS_epoch[~reject_mask]
        EMG_epoch_clean = EMG_epoch[~reject_mask] if EMG is not None else None

        EEG_epoch_car = EEG_epoch_clean - np.mean(EEG_epoch_clean, axis = 2, keepdims = True)
        
        filt_ins = Filtering()
        EEG_epoch_norm = filt_ins.zscore(EEG_epoch_car, mode = 'within_ch')
        RMS_epoch_norm = filt_ins.zscore(RMS_epoch_clean, mode = 'within_ch')
        EMG_epoch_norm = filt_ins.zscore(EMG_epoch_clean, mode = 'within_ch') if EMG is not None else None

        return EEG_epoch_norm, RMS_epoch_norm, EMG_epoch_norm, epochs_overview

    def _extract_EEG_EMG_data(self,
                              path_to_EEG_files : Union[list | Path],
                              path_to_EMG_files : Union[list | Path],
                              EEG_preprocessing_func : Callable,
                              EMG_preprocessing_func : Callable,
                              EMG_config_dict : dict):
        print('-------------------------\n'
              'Process for EEG and EMG data\n'
              '-------------------------\n')
        if isinstance(path_to_EEG_files, Path):
            path_to_EEG_files = [path_to_EEG_files]
        if isinstance(path_to_EMG_files, Path):
            path_to_EMG_files = [path_to_EMG_files]
        
        epochs_overview = []
        all_EEG_data = []
        all_EMG_data = []
        all_RMS_data = []

        for EEG_file, EMG_file in zip(path_to_EEG_files, path_to_EMG_files):

            EEG_df = pd.read_csv(EEG_file)
            EEG_raw = EEG_df.iloc[:, 1:17].to_numpy()
            
            EMG_raw = pd.read_csv(EMG_file).to_numpy()

            # Preprocessing
            EEG_temp, EEG_num_epochs = EEG_preprocessing_func(raw_eeg = EEG_raw)

            RMS_temp, EMG_temp, EMG_num_epochs = EMG_preprocessing_func(
                raw_emg = EMG_raw,
                rms_windowsize = EMG_config_dict['rms_windowsize'],
                rms_stepsize = EMG_config_dict['rms_stepsize'],
                hampel_windowsize = EMG_config_dict['hampel_windowsize'],
                hampel_sigma = EMG_config_dict['hampel_sigma'],
                hampel_plot_option = EMG_config_dict['hampel_plot_option']
            )

            if EEG_num_epochs != EMG_num_epochs:
                raise ValueError(f"Number of epochs mismatch between EEG and EMG for files {EEG_file} and {EMG_file}. EEG epochs: {EEG_num_epochs}, EMG epochs: {EMG_num_epochs}")
                
            all_EEG_data.append(EEG_temp)
            all_RMS_data.append(RMS_temp)
            all_EMG_data.append(EMG_temp) if EMG_config_dict['include_EMG'] else None
            epochs_overview.append(EMG_num_epochs)

        EEG = np.concatenate(all_EEG_data, axis = 0)
        RMS = np.concatenate(all_RMS_data, axis = 0)
        EMG = np.concatenate(all_EMG_data, axis = 0) if EMG_config_dict['include_EMG'] else None

        return EEG, RMS, EMG, epochs_overview
    
    def make_dataset_key(self):
        '''
        A method to append 'subject_ID / experiment_name' to JSON file with empty bad epoch list. 
        This is used for the manual bad epoch rejection function. 
        The user can then fill in the bad epochs for each experiment in the JSON file and the manual rejection function will read the bad epochs from the JSON file and reject the epochs accordingly.
        '''

        for finger in ['middle', 'ring', 'pinky', 'pinchGrip', 'fullGrip']:
            # Define subject ID and fingers to append new bad epoch keys to the JSON file.
            data_file = self.find_flex_files(        
                subjects = 'subject_0',
                modality = 'EEG',
                fingers = finger,
                prefix = 'flex'
            )
            
            manual_dict = {}
            save_dir = self.base_dir / 'manual_bad_epochs.json'

            for file in data_file:
                path = Path(file)

                subject = path.parents[1].name     # Get subject ID
                filename = path.stem               # Get experiment and remove .csv
                
                key = f'{subject}_{filename}'           # Create key in format "subjectID_experiment"
                manual_dict[key] = []   # empty bad epochs

            with open(save_dir, "a") as f:
                json.dump(manual_dict, f, indent=4)

class plot_toolbox():
    def add_markers_to_plot(self, plt_axis, marker_file, stop_markers_at = None):
        """
        Reads marker CSV and adds vertical lines + labels at each time.
        CSV must have columns: time, marker_id, description

        args:
            marker_mode: 
                Set to 'continuous' for adding all markers to plot
                Set to 'epoch' and define number of markers to plot
        """

        for _, row in enumerate(marker_file.values):
            t = row[0] - 3
            marker = row[1]
            desc = {
                10 : 'rest',
                20 : 'contract',
                30 : 'release',
                0  : 'end'
                }
            
            if marker not in desc:
                continue

            # vertical line
            plt_axis.axvline(x=t, color='salmon', linestyle='--', alpha=0.5)

            # label text above line
            plt_axis.text(
                t, plt_axis.set_ylim()[1]*0.9, f'{desc[marker]}',
                rotation=90, color='salmon', ha='left', va='top', fontsize=10
            )
            
            if stop_markers_at is not None:
                if stop_markers_at < t:
                    break

class visualize_EEG(plot_toolbox):
    def __init__(self, fs = 125, trial_period = 9, BCI2a_or_OpenBCI = 'OpenBCI'):
        self.fs = fs
        self.tp = trial_period
        self.toolbox_ins = plot_toolbox()
        if BCI2a_or_OpenBCI == 'OpenBCI':
            self.eeg_ch_names = [
            "Fp1", "Fp2",   # frontal pole
            "C3",  "C4",    # central
            "T5",  "T6",    # temporal (posterior)
            "Cz",  "Pz",    # occipital
            "F7",  "F8",    # temporal (anterior)
            "F3",  "F4",    # frontal
            "T3",  "T4",    # temporal (mid)
            "P3",  "P4"     # parietal
            ]
        if BCI2a_or_OpenBCI == 'BCI2a':
            self.eeg_ch_names = ['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']

    def plot_egg_across_channels(self, eeg : np.ndarray, markers : pd.DataFrame | int, display_window : list | int, ch_list : list = None, channels_per_figure : int = 4, bad_epochs : list | None = None):
        '''
        Plot sequential EEG data with RMS envelope OR
        plot mean epoch EEG data

        :param numpy.nDarray egg: EGG data of shape (samples, channels)
        :param pd.DataFrame markers: Provide markers file to display marker or provide a int for disable markers insert
        :param list display_window: Provide a list of two ints [start, end] in secounds to display period of the sequential data and leave as int to display all
        '''
        if isinstance(eeg, dict):
            raise TypeError('Specify which finger with an np.array object')
        if isinstance(display_window, list):
            if len(display_window) != 2:
                raise ValueError('display_window much have two elements of int')
            eeg = eeg[display_window[0]*self.fs : display_window[1]*self.fs, :].copy()
            stop_marker = display_window[1] - display_window[0]
        elif not isinstance(display_window, int):
            raise TypeError('display_window much be of Type list or int')
        else:
            stop_marker = eeg.shape[0] / self.fs
        if ch_list is None:
            ch_list = list(range(eeg.shape[1]))

        # ---- data info ----
        n_samples, _ = eeg.shape
        time = np.arange(n_samples) / self.fs
        ymax = np.max(eeg[:, ch_list])
        ymin = np.min(eeg[:, ch_list])

        # ---- split channels into pages ----
        for i in range(0, len(ch_list), channels_per_figure):

            page_channels = ch_list[i:i + channels_per_figure]
            n_plot = len(page_channels)

            fig, axs = plt.subplots(
                n_plot, 1,
                figsize=(10, 2.2 * n_plot),
                sharex=True,
                dpi=150
            )

            if n_plot == 1:
                axs = [axs]

            # ---- plot each channel ----
            for ax, ch in zip(axs, page_channels):

                signal = eeg[:, ch]

                if bad_epochs is not None:
                    epoch_samples = self.tp * self.fs

                    for ep in bad_epochs:
                        start_sample = ep * epoch_samples
                        end_sample = (ep + 1) * epoch_samples
                        
                        # Convert to time
                        start_time = start_sample / self.fs
                        end_time = end_sample / self.fs
                        
                        # Only draw if visible in current window
                        if start_time <= time[-1]:
                            ax.axvspan(start_time, end_time,
                                    color='yellow', alpha=0.25)
                                        
                ax.plot(time, signal, linewidth=0.7,
                        label=self.eeg_ch_names[ch], color = 'steelblue')

                ax.set_ylim([ymin, ymax])
                if time[-1] <= self.tp * 10:
                    ax.set_xticks(np.arange(0, time[-1], self.tp // 3))
                else:
                    ax.set_xticks(np.arange(0, time[-1], self.tp))
                ax.set_xlim([0, time[-1] + 0.1])
                ax.set_ylabel("EEG")
                ax.legend(loc='upper right')
                ax.grid(alpha=0.3)

                if isinstance(markers, pd.DataFrame):
                    self.toolbox_ins.add_markers_to_plot(
                        plt_axis=ax,
                        marker_file=markers,
                        stop_markers_at=stop_marker
                    )
        

            axs[-1].set_xlabel("Time (s)")

            fig.suptitle(f'Channels {page_channels}', fontsize=12)
            fig.tight_layout()

            # Plot in full screen
            manager = plt.get_current_fig_manager()
            manager.window.state('zoomed')  # Best option in VS Code
            plt.show()
    
    def plot_mrcp(self, mrcp_index : np.ndarray, mrcp_thumb : np.ndarray, useable_channels : list, fs : int = None):
        '''
        Plot sequential EEG data with RMS envelope OR
        plot mean epoch EEG data

        :param numpy.nDarray egg: EGG data of shape (samples, channels)
        :param pd.DataFrame markers: Provide markers file to display marker or provide a int for disable markers insert
        :param list display_window: Provide a list of two ints [start, end] in secounds to display period of the sequential data and leave as int to display all
        '''
        fs = self.fs if fs is None else fs
        # ---- data info ----
        n_samples, n_ch = mrcp_index.shape

        time = np.arange(n_samples) / fs
        ymax = np.max( (np.max(mrcp_index), np.max(mrcp_thumb)) )
        ymin = np.min( (np.min(mrcp_index), np.min(mrcp_thumb)) )


        fig, axs = plt.subplots(
            n_ch, 1,
            figsize=(10, 6),
            sharex=True,
            dpi=150
        )

        # ---- plot each channel ----
        for i in range(n_ch):
            ax = axs[i]

            Xi = mrcp_index[:, i]
            Xt = mrcp_thumb[:, i]
                                    
            ax.plot(time, Xi, linewidth=0.9, label = 'Index at ' + self.eeg_ch_names[useable_channels[i]], color = 'steelblue')
            ax.plot(time, Xt, linewidth=0.9, label = 'Thumb at ' + self.eeg_ch_names[useable_channels[i]], color = 'orange')

            ax.set_ylim([ymin, ymax])
            ax.set_xlim([0, time[-1]])
            ax.set_xticks(np.arange(0, time[-1]+0.1, 1))

            ax.set_ylabel("Relative amplitude (%)")
            axs[-1].set_xlabel("Time (s)")

            ax.legend(loc = 'upper right')
            ax.grid(alpha=0.3)    

            # vertical line
            ax.axhline(y = 0, color='black', linestyle='-', alpha=0.5)
            ax.axvline(x = 3, color='salmon', linestyle='--', alpha=0.5)
            ax.axvline(x = 6, color='salmon', linestyle='--', alpha=0.5)

            # label text above line
            ax.text(
                3, ax.set_ylim()[1]*0.9, 'Onset',
                rotation=90, color='salmon', ha='left', va='top', fontsize=10
            )
            ax.text(
                6, ax.set_ylim()[1]*0.9, 'Release',
                rotation=90, color='salmon', ha='left', va='top', fontsize=10
            )

        fig.suptitle('Frequency Range 13 - 30 Hz', fontsize=12)
        fig.tight_layout()
        plt.show()

class visualize_EMG():
    def __init__(self, fs = 2000, rms_sampling_window = 200, rms_windows_stepsize = 50,  total_epochs = 90, trial_period = 9):
        self.fs = fs
        self.rsw = rms_sampling_window
        self.rws = rms_windows_stepsize
        self.tp = trial_period
        self.te = total_epochs
        self.toolbox_ins = plot_toolbox()
        self.emg_ch_names = [
        'Channel 1 : Palmaris longus',
        'Channel 2 : Flexor digitorum superficialis',
        'Channel 3 : Flexor pollicis longus',
        ]
    
    def plot_rms_across_channels(self, emg : np.ndarray, rms : np.ndarray, markers : pd.DataFrame | int, display_window : list | int, bad_epochs : list | None = None):
        '''
        Plot sequential EMG data with RMS envelope OR
        plot mean epoch EMG data with mean epoch RMS envelope

        :param numpy.nDarray emg: EMG data of shape (samples, channels)
        :param numpy.nDarray rms: RMS data of shape (samples, channels)
        :param pd.DataFrame markers: Provide markers file to display marker or provide a int for disable markers insert
        :param list display_window: Provide a list of two ints [start, end] in secounds to display period of the sequential data and leave as int to display all
        '''
        # Put ymax and ymin window for the whole dataset
        ymax = max(emg.max(), rms.max())
        ymin = min(emg.min(), rms.min())
        if isinstance(emg, dict) or isinstance(rms, dict):
            raise TypeError('Specify which finger with an np.array object')
        if isinstance(display_window, list):
            if len(display_window) != 2:
                raise ValueError('display_window much have two elements of int')
            real_fs = rms.shape[0] / (self.tp * self.te)                        # Real frequency
            ceil_fs = int( np.ceil(real_fs) )                                   # Get as close to the real fs
            ceil_fs = 125
            emg = emg[display_window[0]*self.fs : display_window[1]*self.fs, :].copy()
            rms = rms[display_window[0]*ceil_fs : display_window[1]*ceil_fs, :].copy()
            stop_marker = display_window[1] - display_window[0]
        elif not isinstance(display_window, int):
            raise TypeError('display_window much be of Type list or int')
        else:
            stop_marker = emg.shape[0] / self.fs
        
        # Size of the data
        n_samp_emg, n_ch = emg.shape
        n_samp_rms, _ = rms.shape
        # PUT ymax and ymin calculation here to reflect the display window only
        if isinstance(display_window, list):
            ymax = max(emg.max(), rms.max())
            ymin = min(emg.min(), rms.min())

        fig, axs = plt.subplots(n_ch, 1, figsize = (20, 8))

        time = np.arange(n_samp_emg) / self.fs
        win_time = (np.arange(n_samp_rms) * (self.rws) + self.rws) / self.fs
        
        for ch in range(n_ch):
            EMG = emg[:, ch]
            RMS = rms[:, ch]
            ax = axs[ch]

            if bad_epochs is not None:
                epoch_samples = self.tp * self.fs

                for ep in bad_epochs:
                    start_sample = ep * epoch_samples
                    end_sample = (ep + 1) * epoch_samples
                    
                    # Convert to time
                    start_time = start_sample / self.fs
                    end_time = end_sample / self.fs
                    
                    # Only draw if visible in current window
                    if start_time <= time[-1]:
                        ax.axvspan(start_time, end_time,
                                color='yellow', alpha=0.25)
                        
            ax.plot(time, EMG, label = 'EMG')
            ax.plot(win_time, RMS, label = 'RMS envelope')

            ax.set_ylim([ymin, ymax])
            if time[-1] <= self.tp * 10:
                ax.set_xticks(np.arange(0, time[-1], self.tp // 3))
            else:
                ax.set_xticks(np.arange(0, time[-1], self.tp))
            ax.set_xlim([0, time[-1]+0.1])

            ax.set_title(f'{self.emg_ch_names[ch]}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Standardized EMG (a.u)')
            ax.legend(loc = 'lower left')

            if isinstance(markers, pd.DataFrame):
                self.toolbox_ins.add_markers_to_plot(plt_axis = ax, marker_file = markers, stop_markers_at = stop_marker)

        fig.tight_layout()
        #plt.savefig('edited_images/EMG_data_analysis/meanEpoch_DataDriftAndNorm_withinCH.png')
        
        # Plot in full screen
        manager = plt.get_current_fig_manager()
        manager.window.state('zoomed')  # Best option in VS Code
        plt.show() 

    def plot_EMG_across_channels(self, emg : np.ndarray, markers : pd.DataFrame | int, display_window : list | int):
        '''
        Plot sequential EMG data OR
        plot mean epoch EMG data with mean epoch

        :param numpy.nDarray emg: EMG data of shape (samples, channels)
        :param pd.DataFrame markers: Provide markers file to display marker or provide a int for disable markers insert
        :param list display_window: Provide a list of two ints [start, end] in secounds to display period of the sequential data and leave as int to display all
        '''
        # Put ymax and ymin window for the whole dataset
        ymax = np.max(emg)
        ymin = np.min(emg)
        if isinstance(emg, dict):
            raise TypeError('Specify which finger with an np.array object')
        if isinstance(display_window, list):
            if len(display_window) != 2:
                raise ValueError('display_window much have two elements of int')
            emg = emg[display_window[0]*self.fs : display_window[1]*self.fs, :].copy()
            stop_marker = display_window[1] - display_window[0]
        elif not isinstance(display_window, int):
            raise TypeError('display_window much be of Type list or int')
        else:
            stop_marker = emg.shape[0] / self.fs
        
        # Size of the data
        n_samp_emg, n_ch = emg.shape
        # PUT ymax and ymin calculation here to reflect the display window only
        if isinstance(display_window, list):
            ymax = np.max(emg)
            ymin = np.min(emg)

        fig, axs = plt.subplots(n_ch, 1, figsize = (20, 8))

        time = np.arange(n_samp_emg) / self.fs
        
        for ch in range(n_ch):
            EMG = emg[:, ch]
            ax = axs[ch]

            ax.plot(time, EMG, label = 'EMG')

            ax.set_ylim([ymin, ymax])
            if time[-1] <= self.tp * 10:
                ax.set_xticks(np.arange(0, time[-1], self.tp // 3))
            else:
                ax.set_xticks(np.arange(0, time[-1], self.tp))
            ax.set_xlim([0, time[-1]+0.1])

            ax.set_title(f'{self.emg_ch_names[ch]}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Standardized EMG (a.u)')
            ax.legend(loc = 'lower left')

            if isinstance(markers, pd.DataFrame):
                self.toolbox_ins.add_markers_to_plot(plt_axis = ax, marker_file = markers, stop_markers_at = stop_marker)

        fig.tight_layout()
        #plt.savefig('edited_images/EMG_data_analysis/meanEpoch_DataDriftAndNorm_withinCH.png')
        
        # Plot in full screen
        manager = plt.get_current_fig_manager()
        manager.window.state('zoomed')  # Best option in VS Code
        plt.show()

def compute_mrcp(epochs, baseline_start = 0.5, baseline_end = 2.5, fs = 125):
    """
    epochs shape: (n_epochs, n_samples, n_channels)

    baseline_samples:
        number of samples BEFORE movement onset
        e.g. 250 samples = 2 s at 125 Hz

    Returns
    -------
    mrcp : (n_samples, n_channels)
    """
    # 0.5 - 2.5 seconds before movement onset = 62.5 - 312.5 samples at 125 Hz
    r0 = int(baseline_start * fs)  # 62.5 samples = 62 samples (rounded down)
    k = int(baseline_end * fs)  # 312.5 samples = 312 samples (rounded down)

    # ---- 1) baseline each trial ----
    baseline = epochs[:, r0:k, :].mean(axis=1, keepdims=True)
    epochs_baselined = epochs - baseline

    # ---- 2) average trials (THIS is MRCP) ----
    mrcp = epochs_baselined.mean(axis=0)

    return mrcp

def compute_erd_ers(eeg_epoch, fs, baseline_start, baseline_end):
    """
    Compute ERD/ERS for EEG data.

    Parameters
    ----------
    eeg_epoch : np.ndarray
        Shape (epochs, sequence, channels)
    fs : int
        Sampling frequency
    band : tuple
        Frequency band (low, high), e.g. (8,13) for mu
    onset_time : float
        Movement onset in seconds (e.g. 3.0)
    baseline_window : tuple
        Relative to onset (e.g. (-1, 0))

    Returns
    -------
    erd_ers : np.ndarray
        Shape (epochs, sequence, channels)
    """
    from scipy.ndimage import uniform_filter1d
    epochs, T, C = eeg_epoch.shape

    # =========================
    # 1. Power (squared signal)
    # =========================
    power = eeg_epoch ** 2

    # Optional smoothing
    power = uniform_filter1d(power, size=int(0.25 * fs), axis=1)

    # =========================
    # 2. Convert time → indices
    # =========================
    baseline_start = int(baseline_start * fs)
    baseline_end   = int(baseline_end * fs)

    baseline_start = max(0, baseline_start)
    baseline_end   = min(T, baseline_end)
    
    # =========================
    # 3. Baseline power
    # =========================
    baseline_power = np.mean(
        power[:, baseline_start:baseline_end, :],
        axis=1,
        keepdims=True
    )  # shape (epochs, 1, channels)

    # =========================
    # 4. ERD/ERS computation
    # =========================
    erd_ers = (power - baseline_power) / (baseline_power + 1e-10) * 100

    return erd_ers

def quick_visulize():
    #-----------#
    # Constants #
    #-----------#
    EMG_FREQ = 2000
    EEG_FREQ = 125
    
    EMG_LOWCUT = 20
    EMG_HIGHCUT = 450
    EEG_LOWCUT = 0.5          # 2          MRCP: 0.05-3 Hz  , Sensorimotor rhythms: 8-30 Hz, 
    EEG_HIGHCUT = 32        # 32

    TRIAL_PERIOD = 9
    TRIM_PERIOD = 3

    RMS_SAMPLING_WINDOW = 500           # 250 ms
    RMS_WINDOW_STEPSIZE = 50            # 25 ms (90 % overlap)

    HAMPEL_WINDOWSIZE = 100
    HAMPEL_SIGMA = 2

    EMG_CONFIG_DICT = {
        'rms_windowsize' : RMS_SAMPLING_WINDOW,
        'rms_stepsize' : RMS_WINDOW_STEPSIZE,
        'hampel_windowsize' : HAMPEL_WINDOWSIZE,
        'hampel_sigma' : HAMPEL_SIGMA,
        'hampel_plot_option' : [False, None],
        'include_EMG' : True
    }

    #------------------------#
    # Select what to inspect #
    #------------------------#
    base_dir = Path().resolve() / 'src/experiment/data'
    
    load_ins = load_datasets(base_dir = base_dir)

    EEG_files = load_ins.find_flex_files(
        subjects = 'subject_0',
        modality = 'EEG',
        fingers = 'index',
        prefix = 'flex'
    )

    EMG_files = load_ins.find_flex_files(
        subjects = 'subject_0',
        modality = 'EMG',
        fingers = 'index',
        prefix = 'flex'
    )

    marker_files = load_ins.find_flex_files(
        subjects = 'subject_0',
        modality = 'Markers',
        fingers = 'index',
        prefix = 'flex'
    )

    #-----------#
    # Load data #
    #-----------#
    
    EEG_ins = EEG_preprocessing(fs = EEG_FREQ, bandpass_lowcut = EEG_LOWCUT, bandpass_highcut = EEG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    EMG_ins = EMG_preprocessing(fs = EMG_FREQ, bandpass_lowcut = EMG_LOWCUT, bandpass_highcut = EMG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)

    SELECT_EXP_DATA = 0         # Numerical integer
    EEG, total_epochs_EEG = load_ins._extract_EEG_data(
        path_to_data_files = EEG_files[:],
        preprocessing_func = EEG_ins.preprocessing_routine
    )

    RMS, EMG, epochs_overview = load_ins._extract_EMG_data(         # Without reject bad epochs
        path_to_data_files = EMG_files[:],
        preprocessing_func = EMG_ins.preprocessing_routine,
        EMG_config_dict = EMG_CONFIG_DICT
    )
    
    print('RMS shape final: ', RMS.shape)
    markers = load_ins.load_datasets_marker(marker_files)

    total_epochs = np.sum(epochs_overview)

    EMG_epoch = EMG.reshape(total_epochs, EMG.shape[0] // total_epochs, 3)
    RMS_epoch = RMS.reshape(total_epochs, RMS.shape[0] // total_epochs, 3)
    EEG_epoch = EEG.reshape(total_epochs, EEG.shape[0] // total_epochs, 16)

    vis_EMG_ins = visualize_EMG(fs = EMG_FREQ, rms_sampling_window = RMS_SAMPLING_WINDOW, rms_windows_stepsize = RMS_WINDOW_STEPSIZE, total_epochs = total_epochs, trial_period = TRIAL_PERIOD)
    vis_EEG_ins = visualize_EEG(fs = EEG_FREQ, trial_period = TRIAL_PERIOD)

    print('RMS epoch shape: ', RMS_epoch.shape, EMG_epoch.shape)
    vis_EMG_ins.plot_rms_across_channels(emg = EMG, rms = RMS, markers = markers, display_window = 0)
    vis_EMG_ins.plot_rms_across_channels(emg = EMG_epoch.mean(axis=0), rms = RMS_epoch.mean(axis = 0), markers = markers, display_window = 0)


    all_ch = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # vis_EEG_ins.plot_egg_across_channels(EEG, markers = 0, display_window = 0, ch_list = all_ch, channels_per_figure=3)
    # vis_EEG_ins.plot_egg_across_channels(EEG_epoch.mean(axis=0), markers = markers, display_window = 0, ch_list = all_ch, channels_per_figure=3)

def test_bad_epochs():
    from scipy.stats import norm
    #-----------#
    # Constants #
    #-----------#
    EMG_FREQ = 2000
    EEG_FREQ = 125
    
    EMG_LOWCUT = 20
    EMG_HIGHCUT = 450
    EEG_LOWCUT = 0.5          # 2          MRCP: 0.05-3 Hz  , Sensorimotor rhythms: 8-30 Hz, 
    EEG_HIGHCUT = 30        # 32

    TRIAL_PERIOD = 9
    TRIM_PERIOD = 3

    RMS_SAMPLING_WINDOW = 500           # 250 ms
    RMS_WINDOW_STEPSIZE = 50            # 25 ms (90 % overlap)

    HAMPEL_WINDOWSIZE = 100
    HAMPEL_SIGMA = 2

    EMG_CONFIG_DICT = {
        'rms_windowsize' : RMS_SAMPLING_WINDOW,
        'rms_stepsize' : RMS_WINDOW_STEPSIZE,
        'hampel_windowsize' : HAMPEL_WINDOWSIZE,
        'hampel_sigma' : HAMPEL_SIGMA,
        'hampel_plot_option' : [False, None],
        'include_EMG' : True
    }
    
    REJECT_CONFIG_DICT = {
        'EEG_epoch_rejection_tolerance' : 6,
        'EMG_epoch_rejection_tolerance' : 6,
        'EEG_ch_acceptance' : 0,
        'EMG_ch_acceptance' : 0
    }
    # Tolerance -> RANGE given [6 : 8]
    # EMG -> RANGE given by [0, 1]
    # EEG all CH -> RANGE given [0 : 3]
    # EEG 6 CH -> RANGE given [0 : 2]


    #------------------------#
    # Select what to inspect #
    #------------------------#
    base_dir = Path().resolve() / 'src/experiment/data'
    
    load_ins = load_datasets(base_dir = base_dir)

    EEG_files = load_ins.find_flex_files(
        subjects = 'subject_0',
        modality = 'EEG',
        fingers = 'index',
        prefix = 'flex'
    )

    EMG_files = load_ins.find_flex_files(
        subjects = 'subject_0',
        modality = 'EMG',
        fingers = 'index',
        prefix = 'flex'
    )

    marker_files = load_ins.find_flex_files(
        subjects = 'subject_0',
        modality = 'Markers',
        fingers = 'fullGrip',
        prefix = 'flex'
    )

    #-----------#
    # Load data #
    #-----------#
    reject_ins = RejectBadEpochs(base_dir = base_dir)
    EEG_ins = EEG_preprocessing(fs = EEG_FREQ, bandpass_lowcut = EEG_LOWCUT, bandpass_highcut = EEG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    EMG_ins = EMG_preprocessing(fs = EMG_FREQ, bandpass_lowcut = EMG_LOWCUT, bandpass_highcut = EMG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)

    SELECT_EXP_DATA = 0        # Numerical integer
    EEG, total_epochs_EEG = load_ins._extract_EEG_data(
        path_to_data_files = EEG_files[SELECT_EXP_DATA],
        preprocessing_func = EEG_ins.preprocessing_routine
    )

    RMS, EMG, epochs_overview = load_ins._extract_EMG_data(         # Without reject bad epochs
        path_to_data_files = EMG_files[SELECT_EXP_DATA],
        preprocessing_func = EMG_ins.preprocessing_routine,
        EMG_config_dict = EMG_CONFIG_DICT
    )   

    markers = load_ins.load_datasets_marker(marker_files)[SELECT_EXP_DATA]

    reject_mask = reject_ins.reject_routine(data_file_per_finger = EEG_files[SELECT_EXP_DATA],
                                            epochs_overview = epochs_overview,
                                            EEG_data = EEG,
                                            RMS_data = RMS,
                                            reject_config_dict = REJECT_CONFIG_DICT,
                                            EEG_useable_channels = None)

    total_epochs = np.sum(epochs_overview)
    bad_epochs = np.where(reject_mask)[0]

    EMG_epoch = EMG.reshape(total_epochs, EMG.shape[0] // total_epochs, 3)
    RMS_epoch = RMS.reshape(total_epochs, RMS.shape[0] // total_epochs, 3)
    EEG_epoch = EEG.reshape(total_epochs, EEG.shape[0] // total_epochs, 16)

    zscore = Filtering(fs = 0).zscore
    # EMG = zscore(EMG)
    RMS = zscore(RMS)

    # vis_EMG_ins = visualize_EMG(fs = EMG_FREQ, rms_sampling_window = RMS_SAMPLING_WINDOW, rms_windows_stepsize = RMS_WINDOW_STEPSIZE, total_epochs = total_epochs, trial_period = TRIAL_PERIOD)
    # vis_EEG_ins = visualize_EEG(fs = EEG_FREQ, trial_period = TRIAL_PERIOD)

    # vis_EMG_ins.plot_rms_across_channels(emg = EMG, rms = RMS, markers = markers, display_window = 0, bad_epochs = bad_epochs)
    # # vis_EMG_ins.plot_rms_across_channels(emg = EMG_epoch.mean(axis=0), rms = RMS_epoch.mean(axis = 0), markers = markers, display_window = 0)

    # all_ch = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # vis_EEG_ins.plot_egg_across_channels(EEG, markers = markers, display_window = 0, ch_list = all_ch, channels_per_figure=3, bad_epochs = bad_epochs)
    # vis_EEG_ins.plot_egg_across_channels(EEG_epoch.mean(axis=0), markers = markers, display_window = 0, ch_list = all_ch, channels_per_figure=3)

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

    def plot_bandpower_heatmaps(self, data, subjects, REGIONS, BANDS):
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

        mean_mat = np.mean(all_mats, axis=0)   # (C, R, B)
        std_mat  = np.std(all_mats, axis=0)    # (C, R, B)

        class_names = ['Rest', 'Contract', 'Release']

        for c in range(C):
            print(f"\nClass: {class_names[c]}")
            print("Mean:\n", np.round(mean_mat[c], 2))
            print("Std:\n", np.round(std_mat[c], 2))

        #======================================#
        # Normalize color scale across classes #
        #======================================#
        fig, axes = plt.subplots(nrows = C, ncols = S, figsize = (4*S, 3*C))
        class_names = ['Rest', 'Contract', 'Release']

        # all_vals = []
        # for mats in class_data.values():
        #     all_vals.append(np.mean(np.array(mats), axis=0))

        vmin = np.min(all_mats)
        vmax = np.max(all_mats)
        print(f"Color scale range: vmin={vmin:.2f}, vmax={vmax:.2f}")

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

    def statistics(self, data, REGIONS, BANDS):
        import pandas as pd
        import statsmodels.api as sm
        from statsmodels.stats.anova import AnovaRM
        from scipy.stats import ttest_rel
        from statsmodels.stats.multitest import multipletests

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

        for r in range(R):
            for b in range(B):
                data = all_mats[:, :, r, b]   # (S, C)

                df = pd.DataFrame({
                    'subject': np.repeat(np.arange(S), C),
                    'condition': np.tile(['Rest', 'Contract', 'Release'], S),
                    'value': data.flatten()
                })

                model = AnovaRM(df, 'value', 'subject', within=['condition'])
                res = model.fit()

                p = res.anova_table['Pr > F'][0]

                if p < 0.05:
                    print(f"Significant: Region {r}, Band {b}, p={p:.4f}")

        p_vals = []
        tests = []
        print()

        for r in range(R):
            for b in range(B):
                data = all_mats[:, :, r, b]

                rest = data[:, 0]
                contract = data[:, 1]
                release = data[:, 2]

                p1 = ttest_rel(rest, contract).pvalue
                p2 = ttest_rel(rest, release).pvalue
                p3 = ttest_rel(contract, release).pvalue

                p_vals.extend([p1, p2, p3])
                tests.extend([(r,b,'R-C'), (r,b,'R-Rl'), (r,b,'C-Rl')])

        # Correct for multiple comparisons
        reject, p_corr, _, _ = multipletests(p_vals, method='fdr_bh')

        for i, rej in enumerate(reject):
            if rej:
                print(f"Significant {tests[i]}: p={p_corr[i]:.4f}")

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
            (0, 1),  # rest vs contract
            (0, 2),  # rest vs release
            (1, 2)   # contract vs release
        ]
        comparison_names = [
        "rest vs contract",
        "rest vs release",
        "contract vs release"
        ]
        
        d_all_subjects = []
        R = len(REGIONS)
        B = len(BANDS)

        for subj in data_classes:

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
    
    def inspect_frequency_ranges(self, subjects : list):
        EEG_FREQ = 125
        EEG_LOWCUT = 0.5
        EEG_HIGHCUT = 60
        REJECT_CONFIG_DICT = {
            'EEG_epoch_rejection_tolerance' : 6,
            'EMG_epoch_rejection_tolerance' : 6,
            'EEG_ch_acceptance' : 0,
            'EMG_ch_acceptance' : 0
        }

        # SUBJECT_NAME = ['subject_0', 'subject_1', 'subject_2', 'subject_3', 'subject_4', 'subject_5', 'subject_6', 'subject_7', 'subject_8', 'subject_9', 'subject_10', 'subject_11']
        FREQ_BANDS = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 60)
        }
        REGIONS = {
        "prefrontal": ['Fp1', 'Fp2'],
        "frontal": ['F3', 'F4', 'F7', 'F8', 'Fz'],
        "central": ['C3', 'C4', 'Cz'],
        "temporal": ['T3', 'T4', 'T5', 'T6'],
        "parietal": ['P3', 'P4']
        }
        NUM_CH = 16

        # SUBJECT_NAME = ['subject_0','subject_1', 'subject_2', 'subject_3', 'subject_4', 'subject_5', 'subject_6', 'subject_7', 'subject_8', 'subject_9', 'subject_10', 'subject_11', 'subject_12', 'subject_13', 'subject_14', 'subject_15', 'subject_16']     
        SUBJECT_NAME = subjects
        all_subject_data = []
        for subj in SUBJECT_NAME:
            #==============#
            # Load dataset #
            #==============#
            X_epoch_index, _ = self.load_EEG_data(subject_name = subj, finger_name = 'index', reject_config_dict = REJECT_CONFIG_DICT, lowcut = EEG_LOWCUT, highcut = EEG_HIGHCUT)
            X_epoch_thumb, _ = self.load_EEG_data(subject_name = subj, finger_name = 'thumb', reject_config_dict = REJECT_CONFIG_DICT, lowcut = EEG_LOWCUT, highcut = EEG_HIGHCUT)

            #===========================#
            # Extract onset period      #
            # Reshape to continous data #
            #===========================#
            X1_rest, X1_con, X1_rel = self.segment_into_periods(epochs = X_epoch_index)     # shape: (E, S, C) -> (E*S, C)
            X2_rest, X2_con, X2_rel = self.segment_into_periods(epochs = X_epoch_thumb)

            X_rest = np.concatenate((X1_rest, X2_rest), axis=0)
            X_con = np.concatenate((X1_con, X2_con), axis=0)
            X_rel = np.concatenate((X1_rel, X2_rel), axis=0)

            LABELS = ['rest', 'contract', 'release']
            
            data_classes = []                   # Container for classes with regions_features

            for data_class in [X_rest, X_con, X_rel]: 

                region_features = self.compute_trialwise_region_psd(
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
        
        return all_subject_data
        '''
        data_ins.statistics(data = all_subject_data, REGIONS = REGIONS, BANDS = FREQ_BANDS)
        
        data_ins.plot_bandpower_heatmaps(data = all_subject_data, subjects=SUBJECT_NAME, REGIONS = REGIONS, BANDS = FREQ_BANDS)
        
        d_mean, d_std, comparison_names = data_ins.compute_multiclass_separability(all_subject_data, REGIONS=REGIONS, BANDS=FREQ_BANDS)
        
        
        all_mats = d_mean
        C, R, B = all_mats.shape
        S = 1
        #======================================#
        # Normalize color scale across classes #
        #======================================#
        fig, axes = plt.subplots(nrows = C, ncols = S, figsize = (8*S, 3*C))        # 4*S, 3*C
        class_names = ['Rest', 'Contract', 'Release']

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
        #         print(f"  {band}: {d:.3f}")'''
        
if __name__ == '__main__':
    # remove_bad_epochs()
    
    # quick_visulize()
    test_bad_epochs()

    # subjects = ['subject_0', 'subject_1']
    # Da_ins = DataAnalysis()
    # Da_ins.inspect_frequency_ranges(subjects = subjects)
    

    # base_dir = Path().resolve() / 'src/experiment/data'
    # load_ins = load_datasets(base_dir=base_dir)
    # load_ins.make_dataset_key()
    
