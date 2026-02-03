# Manage directory
import re
from pathlib import Path
from typing import List, Union
from collections.abc import Callable

# Manage plots
import matplotlib.pyplot as plt

# Manage data
import pandas as pd
import numpy as np

# From own implementations
from src.utilities.preprocessing import EEG_preprocessing, EMG_preprocessing

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
        num_files = len(path_to_data_files)
        print(num_files)

        marker_dict = {}
        file_idx = 0

        for data_file in path_to_data_files:
             marker_dict[file_idx] = pd.read_csv(data_file)
             file_idx += 1

        return marker_dict
    
    def load_datasets_EEG(self,
                          path_to_data_files : Union[list | Path],
                          preprocessing_func : Callable,
                          bandpass_lowcut : int = 2,
                          bandpass_highcut : int = 32,
                          **preprocess_kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        eeg_num_ch = 16
        all_data = []

        for data_file in path_to_data_files:
        
            EEG_df = pd.read_csv(data_file)
            EEG_raw = EEG_df.iloc[:, 1:17].to_numpy()
            EEG_marker_log = EEG_df.iloc[:, -1].to_numpy()
            
            # Preprocessing
            EEG_filt, num_epochs = preprocessing_func(
                raw_eeg = EEG_raw,
                bandpass_lowcut = bandpass_lowcut,
                bandpass_highcut = bandpass_highcut,
                all_markers = EEG_marker_log,
                **preprocess_kwargs)

            all_data.append(EEG_filt)
            total_epochs += num_epochs

        all_data_array = np.array(all_data)
        EEG = all_data_array.reshape(-1, eeg_num_ch)
        print(f"Reshaped data shape: {EEG.shape}")

        EEG_samples_per_epoch = EEG.shape[0] // total_epochs      
        print(f"Samples per epoch: {EEG_samples_per_epoch}")

        EEG_epoch = EEG.reshape(total_epochs, EEG_samples_per_epoch, eeg_num_ch)

        EEG_epoch_mean = EEG_epoch.mean(axis=0)
        print(f"Epoched data shape: {EEG_epoch.shape}")
        print(f"Mean epoch data shape: {EEG_epoch_mean.shape}")

        return EEG, EEG_epoch, EEG_epoch_mean, total_epochs
    
    def load_datasets_EMG(self,
                          path_to_data_files : Union[list | Path],
                          preprocessing_func : Callable,
                           **preprocess_kwargs):
        
        if isinstance(path_to_data_files, Path):
            path_to_data_files = [path_to_data_files]

        all_data = []
        total_epochs = 0
        emg_num_ch = 3

        for data_file in path_to_data_files:
        
            raw_data = pd.read_csv(data_file).to_numpy()
            
            # Preprocessing
            emg_processed, num_epochs = preprocessing_func(
                raw_emg = raw_data,
                **preprocess_kwargs
            )

            all_data.append(emg_processed)
            total_epochs += num_epochs


        all_data_array = np.array(all_data)
        RMS = all_data_array.reshape(-1, emg_num_ch)
        print(f"Reshaped data shape: {RMS.shape}")

        RMS_samples_per_epoch = RMS.shape[0] // total_epochs      
        print(f"Samples per epoch: {RMS_samples_per_epoch}")

        RMS_epoch = RMS.reshape(total_epochs, RMS_samples_per_epoch, emg_num_ch)

        RMS_epoch_mean = RMS_epoch.mean(axis=0)
        print(f"Epoched data shape: {RMS_epoch.shape}")
        print(f"Mean epoch data shape: {RMS_epoch_mean.shape}")

        return RMS, RMS_epoch, RMS_epoch_mean, total_epochs

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

    def plot_egg_across_channels(self, eeg : np.ndarray, markers : pd.DataFrame | int, display_window : list | int):
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

        # Size of the data
        n_samp_emg, n_ch = eeg.shape        
        ymax = np.max(eeg)
        ymin = np.min(eeg)

        rows = 4
        select_ch = 0
        for _ in range(rows):
            fig, axs = plt.subplots(rows, 1, figsize=(3, 2.5*rows), dpi=150)

            time = np.arange(n_samp_emg) / self.fs

            for ch in range(rows):
                #row = ch % rows
                #col = ch // rows
                ax = axs[ch]
                EEG = eeg[:, select_ch]

                ax.plot(time, EEG, label = self.eeg_ch_names[select_ch])

                ax.set_ylim([ymin, ymax])
                ax.set_xticks(np.arange(0, time[-1]+1, self.tp//3))

                ax.set_xlabel('Time (s)')
                ax.set_ylabel('EEG')
                ax.legend(loc = 'lower left')
                ax.grid()

                if isinstance(markers, pd.DataFrame):
                    self.toolbox_ins.add_markers_to_plot(plt_axis = ax, marker_file = markers, stop_markers_at = stop_marker)

                select_ch += 1
                
            fig.tight_layout()
            plt.show() 

class visualize_EMG():
    def __init__(self, fs = 2000, rms_sampling_window = 200, rms_windows_stepsize = 50, which_finger = 'index', num_epochs = 30, total_epochs = 90, trial_period = 9):
        self.fs = fs
        self.rsw = rms_sampling_window
        self.rws = rms_windows_stepsize
        self.key = which_finger
        self.tp = trial_period
        self.ne = num_epochs
        self.te = total_epochs
        self.toolbox_ins = plot_toolbox()
        self.emg_ch_names = [
        'Channel 1 : Palmaris longus',
        'Channel 2 : Flexor digitorum superficialis',
        'Channel 3 : Flexor pollicis longus',
        ]
    
    def plot_rms_across_channels(self, emg : np.ndarray, rms : np.ndarray, markers : pd.DataFrame | int, display_window : list | int):
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
            ceil_fs = 40
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

            ax.plot(time, EMG, label = self.key)
            ax.plot(win_time, RMS, label = 'RMS envelope')

            ax.set_ylim([ymin, ymax])
            ax.set_xticks(np.arange(0, time[-1]+1, self.tp//3))
            ax.set_xlim([0, time[-1]+1])

            ax.set_title(f'{self.emg_ch_names[ch]}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Standardized EMG (a.u)')
            ax.legend(loc = 'lower left')

            if isinstance(markers, pd.DataFrame):
                self.toolbox_ins.add_markers_to_plot(plt_axis = ax, marker_file = markers, stop_markers_at = stop_marker)

        fig.tight_layout()
        #plt.savefig('edited_images/EMG_data_analysis/meanEpoch_DataDriftAndNorm_withinCH.png')
        plt.show() 

def main():
    #-----------#
    # Constants #
    #-----------#
    EMG_FREQ = 2000
    EEG_FREQ = 125
    
    EMG_LOWCUT = 20
    EMG_HIGHCUT = 450
    EEG_LOWCUT = 2
    EEG_HIGHCUT = 32

    EMG_NUM_CH = 3
    EEG_NUM_CH = 16

    NUM_EPOCHS = 30
    # TOTAL_EPOCHS = 35*3
    TRIAL_PERIOD = 8
    TRIM_PERIOD = 3

    RMS_SAMPLING_WINDOW = 200
    RMS_WINDOW_STEPSIZE = 50

    #------------------------#
    # Select what to inspect #
    #------------------------#
    base_dir = Path().resolve() / 'src/experiment/data'
    
    load_ins = load_datasets(base_dir = base_dir)

    EEG_files = load_ins.find_flex_files(
        subjects = 'subject_0',
        modality = 'EEG',
        fingers = 'thumb',
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
    
    EEG_ins = EEG_preprocessing(fs = EEG_FREQ)
    EMG_ins = EMG_preprocessing(fs = EMG_FREQ,
                                bandpass_lowcut = EMG_LOWCUT,
                                bandpass_highcut = EMG_HIGHCUT,
                                num_epochs = NUM_EPOCHS,
                                trial_period = TRIAL_PERIOD,
                                trim_period = TRIM_PERIOD)
    
    
    # EEG, EEG_epoch, EEG_epoch_mean, total_epochs_EEG = load_ins.load_datasets_EEG(
    #     path_to_data_files = EEG_files,
    #     preprocessing_func = EEG_ins.preprocessing_routine,
    #     bandpass_lowcut = EEG_LOWCUT,
    #     bandpass_highcut = EEG_HIGHCUT,
    #     extract_event = 'all'
    # )

    EMG, EMG_epoch, EMG_epoch_mean, total_epochs_EMG = load_ins.load_datasets_EMG(
        path_to_data_files = EMG_files[1],
        preprocessing_func = EMG_ins.preprocessing_routine
    )
    
    RMS, RMS_epoch, RMS_epoch_mean, total_epochs_RMS = load_ins.load_datasets_EMG(
        path_to_data_files = EMG_files[1],
        preprocessing_func = EMG_ins.preprocessing_routine_rms,
        sample_window = RMS_SAMPLING_WINDOW,
        window_stepsize = RMS_WINDOW_STEPSIZE
    )

    markers = load_ins.load_datasets_marker(marker_files)

    # if total_epochs_EEG != total_epochs_EMG or total_epochs_EEG != total_epochs_RMS or total_epochs_EMG != total_epochs_RMS:
    #     raise ValueError('The total amount of epochs if different from EEG, EMG or RMS')
    # else:
    #     total_epochs = total_epochs_EEG


    # #--------------------------------------#
    # # Select how data should be visualized #
    # #--------------------------------------#
    vis_EMG_ins = visualize_EMG(fs = EMG_FREQ,
                                rms_sampling_window = RMS_SAMPLING_WINDOW,
                                rms_windows_stepsize = RMS_WINDOW_STEPSIZE,
                                which_finger = 'unsued',
                                num_epochs = NUM_EPOCHS,
                                total_epochs = total_epochs_EMG,
                                trial_period = TRIAL_PERIOD)
    vis_EEG_ins = visualize_EEG(fs = EEG_FREQ, trial_period = TRIAL_PERIOD)
    
    # vis_EMG_ins.plot_rms_across_channels(emg = EMG_thumb, rms = RMS_thumb, markers = marker_dict[0], display_window = [0, 120])
    # vis_EMG_ins.plot_rms_across_channels(emg = EMG_thumb, rms = RMS_thumb, markers = marker_dict[0], display_window = [120, 240])
    # vis_EMG_ins.plot_rms_across_channels(emg = EMG_thumb, rms = RMS_thumb, markers = marker_dict[1], display_window = [240, 360])
    # vis_EMG_ins.plot_rms_across_channels(emg = EMG_thumb, rms = RMS_thumb, markers = marker_dict[1], display_window = [360, 480])
    # vis_EMG_ins.plot_rms_across_channels(emg = EMG_thumb, rms = RMS_thumb, markers = marker_dict[2], display_window = [480, 600])
    # vis_EMG_ins.plot_rms_across_channels(emg = EMG_thumb, rms = RMS_thumb, markers = marker_dict[2], display_window = [600, 720])

    vis_EMG_ins.plot_rms_across_channels(emg = EMG_epoch_mean, 
                                         rms = RMS_epoch_mean,
                                         markers = markers[0],
                                         display_window = 0)
    
    # vis_EMG_ins.plot_rms_across_channels(emg = EMG_epoch_mean,
    #                                      rms = RMS_epoch_mean,
    #                                      markers = markers[0],
    #                                      display_window = 0)
    
    # vis_EEG_ins.plot_egg_across_channels(EEG_epoch_mean, markers = markers[0], display_window = 0)

if __name__ == '__main__':
    main()