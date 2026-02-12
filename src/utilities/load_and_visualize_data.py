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
from src.utilities.preprocessing import Filtering, EEG_preprocessing, EMG_preprocessing

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
        eeg_num_ch = 16
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

        EEG_samples_per_epoch = EEG.shape[0] // total_epochs      
        print(f"Samples per epoch: {EEG_samples_per_epoch}")

        EEG_epoch = EEG.reshape(total_epochs, EEG_samples_per_epoch, eeg_num_ch)

        EEG_epoch_mean = EEG_epoch.mean(axis=0)
        print(f"Epoched data shape: {EEG_epoch.shape}")
        print(f"Mean epoch data shape: {EEG_epoch_mean.shape}")

        # baseline = EEG_epoch[:, 0:3*125, :].mean(axis = 0)     # 375, 16
        # baseline = baseline.mean(axis = 0)
        # EEG = EEG - baseline

        # EEG_epoch = EEG.reshape(total_epochs, EEG_samples_per_epoch, eeg_num_ch)

        # EEG_epoch_mean = EEG_epoch.mean(axis=0)

        return EEG, EEG_epoch, EEG_epoch_mean, total_epochs
    
    def load_datasets_EMG(self,
                          path_to_data_files : Union[list | Path],
                          preprocessing_func : Callable,
                          rms_windowsize = 200,
                          rms_stepsize = 50,
                          hampel_windowsize = 100,
                          hampel_sigma = 3,
                          hampel_plot_option = [False, None],
                          include_EMG = False):
        print('------------------\n'
            'Process for EMG data\n'
            '------------------\n')
        if isinstance(path_to_data_files, Path):
            path_to_data_files = [path_to_data_files]

        rms_data = []
        emg_data = []
        total_epochs = 0
        emg_num_ch = 3

        for data_file in path_to_data_files:
            print(data_file)
            raw_data = pd.read_csv(data_file).to_numpy()
            
            # Preprocessing
            rms_temp, emg_temp, num_epochs = preprocessing_func(
                raw_emg = raw_data,
                rms_windowsize = rms_windowsize,
                rms_stepsize = rms_stepsize,
                hampel_windowsize = hampel_windowsize,
                hampel_sigma = hampel_sigma,
                hampel_plot_option = hampel_plot_option
            )

            rms_data.append(rms_temp)
            emg_data.append(emg_temp)
            total_epochs += num_epochs
        
        rts_EMG = None
        if include_EMG:
            EMG = np.concatenate(emg_data, axis = 0)
            EMG_samples_per_epoch = EMG.shape[0] // total_epochs
            EMG_epoch = EMG.reshape(total_epochs, EMG_samples_per_epoch, emg_num_ch)
            EMG_epoch_mean = EMG_epoch.mean(axis=0)
            print(f"Epoched EMG shape: {EMG_epoch.shape}")
            rts_EMG = [EMG, EMG_epoch, EMG_epoch_mean]
        
        RMS = np.concatenate(rms_data, axis = 0)
        RMS_samples_per_epoch = RMS.shape[0] // total_epochs
        RMS_epoch = RMS.reshape(total_epochs, RMS_samples_per_epoch, emg_num_ch)
        RMS_epoch_mean = RMS_epoch.mean(axis=0)
        print(f"Epoched RMS shape: {RMS_epoch.shape}")
        
        return [RMS, RMS_epoch, RMS_epoch_mean], rts_EMG, total_epochs

    '''MY METHOD
    def delete_bad_epoch(self, base_dir, subject_id, file_name, bad_epoch_0_index):
        
        #Remove bad epochs from both EEG and EMG datasets.
        
        data_dir = [base_dir / f'{subject_id}/EEG/{file_name}',
                    base_dir / f'{subject_id}/EMG/{file_name}']

        if not all(path.exists() for path in data_dir):
            raise FileNotFoundError(f'In delete_bad_epoch. One or more paths do not exist:\n{data_dir}')
        
        freqs = [125, 2000]
        channels = [16, 3]
        TRIAL_PERIOD = 8
        TRIM_PERIOD = 3
        EXPECTED_NUM_EPOCHS = 30
        BAND_RANGE = {
            freqs[0] : [0.5, 30], 
            freqs[1] : [20, 450]
            }
        bad_epoch = {}

        for i, fs in enumerate(freqs):
            #============================#
            # 1) Read data and filtering #
            #============================#
            if fs == freqs[0]:              # For EEG
                df = pd.read_csv(data_dir[i])
                data = df.iloc[:, 1:17].to_numpy()
            else:                           # For EMG
                data = pd.read_csv(data_dir[i]).to_numpy()
            
            n_samples, _ = data.shape

            filter_ins = Filtering(fs = fs)
            data_notch = filter_ins.notch(data = data, cutoff = 50, Q = 30)
            data_bandpass, _ = filter_ins.butter_bandpass(data = data_notch, lowcut = BAND_RANGE[fs][0], highcut = BAND_RANGE[fs][1], order = 4)

            #===============================#
            # 2) Calculate number of epochs #
            #===============================#
            num_trim_samples = fs * TRIM_PERIOD * 2                     # Total samples from trim period. WHY *2 : Trim egde on both sides
            num_valid_samples = n_samples - num_trim_samples            # Total samples for experimental period
            cal_num_epochs = num_valid_samples / (fs * TRIAL_PERIOD)    # Divide out total samples in sections of trial periods -> Results in number of epochs
            if cal_num_epochs != EXPECTED_NUM_EPOCHS:                   # Inform if epochs is differnet from usual amount. Can happen if bad trials is removed.
                print(f'OBSERVATION - NUM OF EPOCH ({cal_num_epochs}) IS DIFFERNET FROM USUAL {EXPECTED_NUM_EPOCHS}')

            cal_num_epochs = int(np.round(cal_num_epochs))

            #=========#
            # 3) TRIM #
            #=========#       
            trim_idx_102 = fs * TRIM_PERIOD
            trim_idx_201 = (fs * TRIAL_PERIOD * cal_num_epochs) + trim_idx_102        # WHY instead of data[trim : -trim] -> Inconsistency in protocol causes the last batch of data not be included -> Rare but can happen
            data_trim = data_bandpass[trim_idx_102 : trim_idx_201, :]
            print(f"Original shape {data_bandpass.shape}\n"
                f"Trim 102 idx {trim_idx_102}\n"
                f"Trim 201 idx {trim_idx_201}\n"
                f'EEG_trim shape: {data_trim.shape}\n')
            
            #=====================#
            # 4) Select bad epoch #
            #=====================#
            samples_per_epoch = fs * TRIAL_PERIOD
            epoch = data_trim.reshape(cal_num_epochs, samples_per_epoch, channels[i])
            bad_epoch[fs] = epoch[bad_epoch_0_index, :, :]
        
        EEG_vis_ins = visualize_EEG(fs = freqs[0], trial_period = TRIAL_PERIOD)
        EMG_vis_ins = visualize_EMG(freqs[1], 200, 50, 'non', num_epochs = cal_num_epochs, total_epochs = cal_num_epochs, trial_period = TRIAL_PERIOD)

        EEG_vis_ins.plot_egg_across_channels(bad_epoch[freqs[0]], 0, 0)
        EMG_vis_ins.plot_EMG_across_channels(bad_epoch[freqs[1]], 0, 0)

        msg = input("Do you want to DELETE the epoch. It can't be undone. YES or NO")
        if msg == 'YES':
            # remove epoch from df
            # remove epoch from EMG data
            # save data at the same path
            pass
        elif msg == 'NO':
            pass                
        else:
            raise ValueError(f'In delete_bad_epoch - Input message "{msg}" must be either YES/NO')'''

    def delete_bad_epoch(self, base_dir, subject_id, file_name, bad_epoch_0_index):

        data_dir = [
            base_dir / f'{subject_id}/EEG/{file_name}',
            base_dir / f'{subject_id}/EMG/{file_name}'
        ]

        if not all(path.exists() for path in data_dir):
            raise FileNotFoundError(
                f'In delete_bad_epoch. One or more paths do not exist:\n{data_dir}'
            )

        freqs = [125, 2000]
        TRIAL_PERIOD = 9
        TRIM_PERIOD = 3
        BAND_RANGE = {
            freqs[0] : [0.5, 30], 
            freqs[1] : [20, 450]
        }

        for i, fs in enumerate(freqs):

            df = pd.read_csv(data_dir[i])

            # ----------------------------------
            # Extract signal columns only
            # ----------------------------------
            if fs == 125:
                signal_cols = df.columns[1:17]   # EEG channels
            else:
                signal_cols = df.columns         # EMG only signals

            data = df[signal_cols].to_numpy()
            n_samples = len(df)

            # ----------------------------------
            # Filtering
            # ----------------------------------
            filter_ins = Filtering(fs = fs)
            data_notch = filter_ins.notch(data = data, cutoff = 50, Q = 30)
            data_bandpass, _ = filter_ins.butter_bandpass(data = data_notch, lowcut = BAND_RANGE[fs][0], highcut = BAND_RANGE[fs][1], order = 4)


            # ----------------------------------
            # Calculate trimming indices
            # ----------------------------------
            trim_samples = fs * TRIM_PERIOD
            samples_per_epoch = fs * TRIAL_PERIOD

            valid_samples = n_samples - 2 * trim_samples
            num_epochs = valid_samples // samples_per_epoch

            trim_start = trim_samples
            trim_end = trim_start + num_epochs * samples_per_epoch

            if valid_samples % samples_per_epoch != 0:
                print("Warning: Samples not perfectly divisible by trial period"
                      f"Calculated num epochs: {valid_samples / samples_per_epoch}")
            # ----------------------------------
            # Compute epoch index boundaries
            # ----------------------------------
            epoch_start = trim_start + bad_epoch_0_index * samples_per_epoch
            epoch_end   = epoch_start + samples_per_epoch

            print(f"Deleting samples {epoch_start}:{epoch_end} (fs={fs})")

            if fs == freqs[0]:
                EEG_vis_ins = visualize_EEG(fs = freqs[0], trial_period = TRIAL_PERIOD)
                EEG_vis_ins.plot_egg_across_channels(data_bandpass[epoch_start:epoch_end, :], 0, 0)
                df_clean_EEG = df.drop(index=range(epoch_start, epoch_end)).reset_index(drop=True)
            
            elif fs == freqs[1]:
                EMG_vis_ins = visualize_EMG(freqs[1], 200, 50, total_epochs = num_epochs, trial_period = TRIAL_PERIOD)
                EMG_vis_ins.plot_EMG_across_channels(data_bandpass[epoch_start:epoch_end, :], 0, 0)
                df_clean_EMG = df.drop(index=range(epoch_start, epoch_end)).reset_index(drop=True)

        msg = input("Do you want to DELETE the epoch. It can't be undone. YES or NO\nINPUT: ")
        if str.upper(msg) == 'YES':
            for i in range(len(data_dir)):
                df_clean_EEG.to_csv('EEG.csv', index=False)
                df_clean_EMG.to_csv('EMG.csv', index=False)
        elif str.upper(msg) == 'NO':
            pass                
        else:
            raise ValueError(f'In delete_bad_epoch - Input message "{msg}" must be either YES/NO')

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
            fig, axs = plt.subplots(rows, 1, figsize=(10, 2.5*rows), dpi=150)

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

            ax.plot(time, EMG, label = 'EMG')
            ax.plot(win_time, RMS, label = 'RMS envelope')

            ax.set_ylim([ymin, ymax])
            ax.set_xticks(np.arange(0, time[-1]+1, self.tp//3))
            ax.set_xlim([0, time[-1]+0.1])

            ax.set_title(f'{self.emg_ch_names[ch]}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Standardized EMG (a.u)')
            ax.legend(loc = 'lower left')

            if isinstance(markers, pd.DataFrame):
                self.toolbox_ins.add_markers_to_plot(plt_axis = ax, marker_file = markers, stop_markers_at = stop_marker)

        fig.tight_layout()
        #plt.savefig('edited_images/EMG_data_analysis/meanEpoch_DataDriftAndNorm_withinCH.png')
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
            ax.set_xticks(np.arange(0, time[-1]+1, self.tp//3))
            ax.set_xlim([0, time[-1]+0.1])

            ax.set_title(f'{self.emg_ch_names[ch]}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Standardized EMG (a.u)')
            ax.legend(loc = 'lower left')

            if isinstance(markers, pd.DataFrame):
                self.toolbox_ins.add_markers_to_plot(plt_axis = ax, marker_file = markers, stop_markers_at = stop_marker)

        fig.tight_layout()
        #plt.savefig('edited_images/EMG_data_analysis/meanEpoch_DataDriftAndNorm_withinCH.png')
        plt.show() 

def remove_bad_epochs():
    base_dir = Path().resolve() / 'src/experiment/data'
    load_ins = load_datasets(base_dir = base_dir)

    subject_id = 'subject_4'
    file_name = 'flex_thumb_finger_2026-02-12 10-19-38.csv'
    bad_epoch_0_index = 25

    load_ins.delete_bad_epoch(base_dir = base_dir,
                              subject_id = subject_id,
                              file_name = file_name,
                              bad_epoch_0_index = bad_epoch_0_index)

def main():
    #-----------#
    # Constants #
    #-----------#
    EMG_FREQ = 2000
    EEG_FREQ = 125
    
    EMG_LOWCUT = 20
    EMG_HIGHCUT = 450
    EEG_LOWCUT = 0.3          # 2
    EEG_HIGHCUT = 30        # 32

    TRIAL_PERIOD = 9
    TRIM_PERIOD = 3

    RMS_SAMPLING_WINDOW = 200
    RMS_WINDOW_STEPSIZE = 50

    HAMPEL_WINDOWSIZE = 100
    HAMPEL_SIGMA = 2

    #------------------------#
    # Select what to inspect #
    #------------------------#
    base_dir = Path().resolve() / 'src/experiment/data'
    
    load_ins = load_datasets(base_dir = base_dir)

    EEG_files = load_ins.find_flex_files(
        subjects = 'subject_4',
        modality = 'EEG',
        fingers = 'index',
        prefix = 'flex'
    )

    EMG_files = load_ins.find_flex_files(
        subjects = 'subject_5',
        modality = 'EMG',
        fingers = 'thumb',
        prefix = 'flex'
    )

    marker_files = load_ins.find_flex_files(
        subjects = 'subject_5',
        modality = 'Markers',
        fingers = 'thumb',
        prefix = 'flex'
    )

    #-----------#
    # Load data #
    #-----------#
    
    EEG_ins = EEG_preprocessing(fs = EEG_FREQ, bandpass_lowcut = EEG_LOWCUT, bandpass_highcut = EEG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    EMG_ins = EMG_preprocessing(fs = EMG_FREQ,
                                bandpass_lowcut = EMG_LOWCUT,
                                bandpass_highcut = EMG_HIGHCUT,
                                trial_period = TRIAL_PERIOD,
                                trim_period = TRIM_PERIOD)
    
    
    # EEG, EEG_epoch, EEG_epoch_mean, total_epochs_EEG = load_ins.load_datasets_EEG(
    #     path_to_data_files = EEG_files,
    #     preprocessing_func = EEG_ins.preprocessing_routine
    # )

    RMS_list, EMG_list, total_epochs_EMG = load_ins.load_datasets_EMG(
        path_to_data_files = EMG_files,
        preprocessing_func = EMG_ins.preprocessing_routine,
        rms_windowsize = RMS_SAMPLING_WINDOW,
        rms_stepsize = RMS_WINDOW_STEPSIZE,
        hampel_windowsize = HAMPEL_WINDOWSIZE,
        hampel_sigma = HAMPEL_SIGMA,
        hampel_plot_option = [False, None],
        include_EMG = True
    )

    RMS, RMS_epoch, RMS_epoch_mean = RMS_list
    EMG, EMG_epoch, EMG_epoch_mean = EMG_list

    markers = load_ins.load_datasets_marker(marker_files)

    for epo in range(RMS_epoch.shape[0]):
        maxpeak = np.max(RMS_epoch[epo, :, :])
        if maxpeak > 10:
            print("BAD EPOCH:", epo, maxpeak)
    #--------------------------------------#
    # Select how data should be visualized #
    #--------------------------------------#
    vis_EEG_ins = visualize_EEG(fs = EEG_FREQ, trial_period = TRIAL_PERIOD)
    vis_EMG_ins = visualize_EMG(fs = EMG_FREQ,
                            rms_sampling_window = RMS_SAMPLING_WINDOW,
                            rms_windows_stepsize = RMS_WINDOW_STEPSIZE,
                            total_epochs = total_epochs_EMG,
                            trial_period = TRIAL_PERIOD)

    # vis_EEG_ins.plot_egg_across_channels(EEG_epoch_mean, markers = markers[0], display_window = 0)

    vis_EMG_ins.plot_rms_across_channels(emg = EMG, 
                                         rms = RMS,
                                         markers = markers[0],
                                         display_window = 0)
    
    vis_EMG_ins.plot_rms_across_channels(emg = EMG_epoch_mean,
                                         rms = RMS_epoch_mean,
                                         markers = markers[0],
                                         display_window = 0)

if __name__ == '__main__':
    #remove_bad_epochs()
    main()
    
