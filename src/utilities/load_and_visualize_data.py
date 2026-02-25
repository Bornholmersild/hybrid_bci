# Manage directory
import re
from pathlib import Path
from typing import List, Union
from collections.abc import Callable

# Manage plots
from flask import json
import matplotlib.pyplot as plt

# Manage data
import pandas as pd
import numpy as np

# From own implementations
from src.utilities.preprocessing import Filtering, EEG_preprocessing, EMG_preprocessing, RejectBadEpochs
# from src.utilities.EEG_feature_extraction import FeatureExtraction

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
    
    def load_datasets_EEG(self,
                          path_to_data_files : Union[list | Path],
                          preprocessing_func : Callable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Load EEG data set given desired data path files from -> find_flex_files

        Parameters
        ----------
        path_to_data_files : list
            A list of pathways to data files
        bandpass_lowcut : int
            Desired lowcut bandpass frequency
        bandpass_highcut : int
            Desired highcut bandpass frequency
        extract_event : str
            Extract period of events (For example: 'all', 'contract', 'release', 'rest') by default 'all'

        Returns
        -------
        :return: Continues preprocessed EEG data
        :return: Epoch preprocessed EEG data
        :return: Mean epoch preprocessed EEG data
        '''
        if isinstance(path_to_data_files, Path):
            path_to_data_files = [path_to_data_files]
        print('------------------\n'
              'Process for EEG data\n'
              '------------------\n')

        total_epochs = 0
        all_data = []

        for data_file in path_to_data_files:
            print(data_file)
        
            EEG_df = pd.read_csv(data_file)
            EEG_raw = EEG_df.iloc[:, 1:17].to_numpy()
            #EEG_marker_log = EEG_df.iloc[:, -1].to_numpy()
            
            # Preprocessing
            EEG_filt, num_epochs = preprocessing_func(raw_eeg = EEG_raw)

            all_data.append(EEG_filt)
            total_epochs += num_epochs

        EEG = np.concatenate(all_data, axis = 0)
        print(f"Reshaped data shape: {EEG.shape}")

        return EEG, total_epochs
    
    def load_datasets_EMG(self,
                          path_to_data_files : Union[list | Path],
                          preprocessing_func : Callable,
                          EMG_config_dict : dict):
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

    def load_datasets(self, 
                      path_to_EEG_files : Union[list | Path],
                      path_to_EMG_files : Union[list | Path],
                      EEG_preprocessing_func : Callable,
                      EMG_preprocessing_func : Callable,
                      EMG_config_dict : dict) -> tuple[list, list, list, int]:
        
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
        base_dir = Path().resolve() / 'src/experiment/data'
        load_ins = load_datasets(base_dir = base_dir)

        # Define subject ID and fingers to append new bad epoch keys to the JSON file.
        data_file = load_ins.find_flex_files(        
            subjects = ['WHO'],
            modality = 'EEG',
            fingers = 'thumb',
            prefix = 'flex'
        )
        
        manual_dict = {}
        save_dir = base_dir / 'manual_bad_epochs.json'

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
    def __init__(self, fs = 125, trial_period = 9):
        self.fs = fs
        self.tp = trial_period
        self.toolbox_ins = plot_toolbox()
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

def quick_visulize():
    #-----------#
    # Constants #
    #-----------#
    EMG_FREQ = 2000
    EEG_FREQ = 125
    
    EMG_LOWCUT = 20
    EMG_HIGHCUT = 450
    EEG_LOWCUT = 2          # 2          MRCP: 0.05-3 Hz  , Sensorimotor rhythms: 8-30 Hz, 
    EEG_HIGHCUT = 32        # 32

    TRIAL_PERIOD = 9
    TRIM_PERIOD = 3

    RMS_SAMPLING_WINDOW = 32
    RMS_WINDOW_STEPSIZE = 16

    HAMPEL_WINDOWSIZE = 100
    HAMPEL_SIGMA = 2

    EMG_CONFIG_DICT = {
        'rms_windowsize' : 32,
        'rms_stepsize' : 16,
        'hampel_windowsize' : 100,
        'hampel_sigma' : 2,
        'hampel_plot_option' : [False, None],
        'include_EMG' : True
    }

    #------------------------#
    # Select what to inspect #
    #------------------------#
    base_dir = Path().resolve() / 'src/experiment/data'
    
    load_ins = load_datasets(base_dir = base_dir)

    EEG_files = load_ins.find_flex_files(
        subjects = 'subject_9',
        modality = 'EEG',
        fingers = 'thumb',
        prefix = 'flex'
    )

    EMG_files = load_ins.find_flex_files(
        subjects = 'subject_9',
        modality = 'EMG',
        fingers = 'thumb',
        prefix = 'flex'
    )

    marker_files = load_ins.find_flex_files(
        subjects = 'subject_9',
        modality = 'Markers',
        fingers = 'thumb',
        prefix = 'flex'
    )

    #-----------#
    # Load data #
    #-----------#
    
    EEG_ins = EEG_preprocessing(fs = EEG_FREQ, bandpass_lowcut = EEG_LOWCUT, bandpass_highcut = EEG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    EMG_ins = EMG_preprocessing(fs = EMG_FREQ, bandpass_lowcut = EMG_LOWCUT, bandpass_highcut = EMG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)

    SELECT_EXP_DATA = 0         # Numerical integer
    EEG,  total_epochs_EEG = load_ins.load_datasets_EEG(
        path_to_data_files = EEG_files,
        preprocessing_func = EEG_ins.preprocessing_routine
    )

    RMS, EMG, epochs_overview = load_ins.load_datasets_EMG(
        path_to_data_files = EMG_files,
        preprocessing_func = EMG_ins.preprocessing_routine,
        EMG_config_dict = EMG_CONFIG_DICT
    )

    markers = load_ins.load_datasets_marker(marker_files)

    total_epochs_EMG = np.sum(epochs_overview)

    EMG_epoch = EMG.reshape(total_epochs_EMG, EMG.shape[0] // total_epochs_EMG, 3)
    RMS_epoch = RMS.reshape(total_epochs_EMG, RMS.shape[0] // total_epochs_EMG, 3)
    EEG_epoch = EEG.reshape(total_epochs_EEG, EEG.shape[0] // total_epochs_EEG, 16)

    vis_EMG_ins = visualize_EMG(fs = EMG_FREQ, rms_sampling_window = RMS_SAMPLING_WINDOW, rms_windows_stepsize = RMS_WINDOW_STEPSIZE, total_epochs = total_epochs_EMG, trial_period = TRIAL_PERIOD)
    vis_EEG_ins = visualize_EEG(fs = EEG_FREQ, trial_period = TRIAL_PERIOD)

    vis_EMG_ins.plot_rms_across_channels(emg = EMG, rms = RMS, markers = markers, display_window = 0)
    vis_EMG_ins.plot_rms_across_channels(emg = EMG_epoch.mean(axis=0), rms = RMS_epoch.mean(axis = 0), markers = markers, display_window = 0)

    all_ch = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # vis_EEG_ins.plot_egg_across_channels(EEG, markers = markers, display_window = 0, ch_list = all_ch, channels_per_figure=3)
    vis_EEG_ins.plot_egg_across_channels(EEG_epoch.mean(axis=0), markers = markers, display_window = 0, ch_list = all_ch, channels_per_figure=3)

def test_bad_epochs():
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
        'EMG_epoch_rejection_tolerance' : 6,
        'EEG_ch_acceptance' : 1,
        'EMG_ch_acceptance' : 1
    }
    # Tolerance -> RANGE given [6 : 8]
    # EMG -> RANGE given by [0, 1]
    # EEG all CH -> RANGE given [0 : 3]
    # EEG 6 CH -> RANGE given [0 : 2]

    base_dir = Path().resolve() / 'src/experiment/data'
    load_ins = load_datasets(base_dir = base_dir)

    EEG_files = load_ins.find_flex_files(
        subjects = 'subject_6',
        modality = 'EEG',
        fingers = 'thumb',
        prefix = 'flex'
    )

    EMG_files = load_ins.find_flex_files(
        subjects = 'subject_6',
        modality = 'EMG',
        fingers = 'thumb',
        prefix = 'flex'
    )

    marker_files = load_ins.find_flex_files(
        subjects = 'subject_7',
        modality = 'Markers',
        fingers = 'thumb',
        prefix = 'flex'
    )

    EEG_ins = EEG_preprocessing(fs = 125, bandpass_lowcut = 2, bandpass_highcut = 32, trial_period = 9, trim_period = 3)
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
    
    markers = load_ins.load_datasets_marker(marker_files)

    reject_mask = reject_ins.reject_routine(data_file_per_finger = EEG_files[SELECT_EXP],
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

    filt_ins = Filtering()
    EEG = filt_ins.zscore(EEG, mode = 'within_ch')
    EMG = filt_ins.zscore(EMG, mode = 'within_ch')
    RMS = filt_ins.zscore(RMS, mode = 'within_ch')

    vis_EEG_ins = visualize_EEG(fs = EEG_FREQ, trial_period = TRIAL_PERIOD)
    vis_EMG_ins = visualize_EMG(2000, 32, 16, total_epochs, 9)

    vis_EMG_ins.plot_rms_across_channels(EMG, RMS, markers, display_window=0, bad_epochs = reject_mask_indices)

    all_ch = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # vis_EEG_ins.plot_egg_across_channels(EEG, markers = markers, display_window = 0, ch_list = all_ch, channels_per_figure=3, bad_epochs = reject_mask_indices)
    
    # filt_ins = Filtering()
    # EEG_epoch = filt_ins.zscore(EEG_epoch, mode = 'within_ch')
    # EEG_epoch_clean = filt_ins.zscore(EEG_epoch_clean, mode = 'within_ch')
    # EMG_epoch = filt_ins.zscore(EMG_epoch, mode = 'within_ch')
    # EMG_epoch_clean = filt_ins.zscore(EMG_epoch_clean, mode = 'within_ch')
    # RMS_epoch = filt_ins.zscore(RMS_epoch, mode = 'within_ch')
    # RMS_epoch_clean = filt_ins.zscore(RMS_epoch_clean, mode = 'within_ch')

    # vis_EMG_ins.plot_rms_across_channels(EMG_epoch.mean(axis = 0), RMS_epoch.mean(axis = 0), markers, display_window=0)
    # vis_EMG_ins.plot_rms_across_channels(EMG_epoch_clean.mean(axis = 0), RMS_epoch_clean.mean(axis = 0), markers, display_window=0)

    # vis_EEG_ins.plot_egg_across_channels(EEG_epoch.mean(axis = 0), markers = markers, display_window = 0, ch_list = all_ch, channels_per_figure=3)
    # vis_EEG_ins.plot_egg_across_channels(EEG_epoch_clean.mean(axis = 0), markers = markers, display_window = 0, ch_list = all_ch, channels_per_figure=3)
    
    '''
    EEG_ins = EEG_preprocessing(fs = 125, bandpass_lowcut = 0.05, bandpass_highcut = 32, trial_period = 9, trim_period = 3)
    EMG_ins = EMG_preprocessing(fs = 2000, bandpass_lowcut = 20, bandpass_highcut = 450, trial_period = 9, trim_period = 3)

    SELECT_EXP_DATA = 2         # Numerical integer
    EEG, EEG_epoch, EEG_epoch_mean, total_epochs_EEG = load_ins.load_datasets_EEG(
        path_to_data_files = EEG_files[0],
        preprocessing_func = EEG_ins.preprocessing_routine
    )

    RMS_list, EMG_list, total_epochs_EMG = load_ins.load_datasets_EMG(
        path_to_data_files = EMG_files[0],
        preprocessing_func = EMG_ins.preprocessing_routine,
        rms_windowsize = 32,
        rms_stepsize = 16,
        hampel_windowsize = 100,
        hampel_sigma = 2,
        hampel_plot_option = [False, None],
        include_EMG = True
    )
    RMS, RMS_epoch, RMS_epoch_mean = RMS_list
    EMG, EMG_epoch, EMG_epoch_mean = EMG_list

    markers = load_ins.load_datasets_marker(marker_files)

    print("----- For EEG ------")
    EEG_bad_mask = detect_bad_epochs_ptp(EEG_epoch, k = 7)
    EEG_bad_indices = np.where(EEG_bad_mask)[0]
    
    print("----- For EMG ------")
    EMG_bad_mask = detect_bad_epochs_ptp(RMS_epoch, k = 6)
    EMG_bad_indices = np.where(EMG_bad_mask)[0]

    vis_EEG_ins = visualize_EEG(fs = EEG_FREQ, trial_period = TRIAL_PERIOD)
    vis_EMG_ins = visualize_EMG(2000, 32, 16, total_epochs_EMG, 9)

    vis_EMG_ins.plot_rms_across_channels(EMG, RMS, markers, display_window=0, bad_epochs = EMG_bad_indices)

    all_ch = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    vis_EEG_ins.plot_egg_across_channels(EEG, markers = markers, display_window = 0, ch_list = all_ch, channels_per_figure=3, bad_epochs=EEG_bad_indices)
    '''

if __name__ == '__main__':
    # remove_bad_epochs()
    quick_visulize()
    # test_bad_epochs()
    
