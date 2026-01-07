# Manage directory
import re
from pathlib import Path

# Manage plots
import matplotlib.pyplot as plt

# Manage data
import pandas as pd
import numpy as np

# From own implementations
from src.utilities.preprocessing import EEG_preprocessing, EMG_preprocessing

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
            t = row[0]
            marker = row[1]
            desc = {
                10 : 'rest',
                20 : 'contract',
                30 : 'release',
                0  : 'end'
                }

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
    def __init__(self, fs = 125, which_finger = 'index', num_epochs = 32, trial_period = 9):
        self.fs = fs
        self.key = which_finger
        self.ne = num_epochs
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

        rows = int(np.ceil(n_ch / 2))
        fig, axs = plt.subplots(rows, 2, figsize=(6, 2.5*rows), dpi=150)

        time = np.arange(n_samp_emg) / self.fs

        for ch in range(n_ch):
            row = ch % rows
            col = ch // rows
            ax = axs[row, col]
            EEG = eeg[:, ch]

            ax.plot(time, EEG, label = self.eeg_ch_names[ch])

            ax.set_ylim([ymin, ymax])
            ax.set_xticks(np.arange(0, time[-1]+1, self.tp//3))

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('EEG')
            ax.legend(loc = 'lower left')

            if isinstance(markers, pd.DataFrame):
                self.toolbox_ins.add_markers_to_plot(plt_axis = ax, marker_file = markers, stop_markers_at = stop_marker)

        fig.tight_layout()
        plt.show() 

class visualize_EMG():
    def __init__(self, fs = 2000, rms_sampling_window = 200, rms_windows_stepsize = 50, which_finger = 'index', num_epochs = 32, trial_period = 9):
        self.fs = fs
        self.rsw = rms_sampling_window
        self.rws = rms_windows_stepsize
        self.key = which_finger
        self.tp = trial_period
        self.ne = num_epochs
        self.toolbox_ins = plot_toolbox()
        self.emg_ch_names = [
        'Channel 1 : Palmaris longus',
        'Channel 2 : Flexor digitorum superficialis',
        'Channel 3 : Flexor pollicis longus',
        'Channel 4 : ',
        'Channel 5 : ',
        'Channel 6 : '
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
        if isinstance(emg, dict) or isinstance(rms, dict):
            raise TypeError('Specify which finger with an np.array object')
        if isinstance(display_window, list):
            if len(display_window) != 2:
                raise ValueError('display_window much have two elements of int')
            real_rms_fs = int(rms.shape[0] / (self.tp * self.ne))
            emg = emg[display_window[0]*self.fs : display_window[1]*self.fs, :].copy()
            rms = rms[display_window[0]*real_rms_fs : display_window[1]*real_rms_fs, :].copy()
            stop_marker = display_window[1] - display_window[0]
        elif not isinstance(display_window, int):
            raise TypeError('display_window much be of Type list or int')
        else:
            stop_marker = emg.shape[0] / self.fs
        
        # Size of the data
        n_samp_emg, n_ch = emg.shape
        n_samp_rms, _ = rms.shape
        ymax = np.max(emg)
        ymin = np.min(emg)

        fig, axs = plt.subplots(n_ch, 1)

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

            ax.set_title(f'{self.emg_ch_names[ch]}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Standardized EMG (a.u)')
            ax.legend(loc = 'lower left')

            if isinstance(markers, pd.DataFrame):
                self.toolbox_ins.add_markers_to_plot(plt_axis = ax, marker_file = markers, stop_markers_at = stop_marker)

        fig.tight_layout()
        plt.show() 

def load_multi_csv(csv_files, data_type=None, which_motion = 'flex'):
    dfs = []
    for f in csv_files:
        # Extract finger name only
        match = re.search(fr'{which_motion}_(.*?)_finger_', f.stem)
        finger = match.group(1) if match else 'unknown'                

        if data_type == 'EEG':
            df = pd.read_csv(f, delimiter=',')
        else:
            df = pd.read_csv(f)
        
        df['finger'] = finger
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

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

    EMG_NUM_CH = 2
    EEG_NUM_CH = 16

    NUM_EPOCHS = 10
    TRIAL_PERIOD = 9

    RMS_SAMPLING_WINDOW = 200
    RMS_WINDOW_STEPSIZE = 50

    #------------------------#
    # Select what to inspect #
    #------------------------#
    choise_subject_file = 'Nicklas_basic_movement'           # Lies in experiment\data\...
    which_motion = 'flex'           # flex, con or * for all
    which_finger = 'index'          # thumb, index, middle, ring, little, or * for all

    DATA_DIR = str(Path().resolve()) + '/src/experiment/data'       # Path to data
    SF = f'{which_motion}_{which_finger}_finger_*.csv'              # What to look for

    #-----------#
    # Load data #
    #-----------#
    EMG_data_dir = Path(f'{DATA_DIR}/{choise_subject_file}/EMG')
    csv_files = list(EMG_data_dir.glob(SF))          # Find path to all files of this name
    EMG_raw_df = load_multi_csv(csv_files)
    
    EEG_data_dir = Path(f'{DATA_DIR}/{choise_subject_file}/EEG')
    csv_files = list(EEG_data_dir.glob(SF))          # Find path to all files of this name
    EEG_raw_df = load_multi_csv(csv_files, data_type='EEG')

    # For markers
    markers_data_dir = Path(f'{DATA_DIR}/{choise_subject_file}/Markers')
    csv_files = list(markers_data_dir.glob(SF))          # Find path to all files of this name
    markers_df = load_multi_csv(csv_files)

    #---------------------------#
    # Extract the data channels #
    #---------------------------#
    EMG_raw = {
        f: EMG_raw_df.loc[EMG_raw_df['finger'] == f, EMG_raw_df.columns.str.startswith('ch')].to_numpy()
        for f in EMG_raw_df['finger'].unique()
    }

    EEG_raw = {
        f: EEG_raw_df.loc[EEG_raw_df['finger'] == f, EEG_raw_df.columns.str.startswith(' EXG')].to_numpy()
        for f in EEG_raw_df['finger'].unique()
    }

    markers = {
        f: markers_df.loc[markers_df['finger'] == f]
        for f in EEG_raw_df['finger'].unique()
    }

    #------------------------------------#
    # Perform data preprocessing routine #
    #------------------------------------#
    EMG_pre_ins = EMG_preprocessing(fs = EMG_FREQ)
    EEG_pre_ins = EEG_preprocessing(fs = EEG_FREQ)

    RMS, RMS_epoch, RMS_epoch_mean = EMG_pre_ins.preprocessing_routine_rms(raw_emg = EMG_raw,
                                                                        bandpass_lowcut = EMG_LOWCUT,
                                                                        bandpass_highcut = EMG_HIGHCUT,
                                                                        num_channels = EMG_NUM_CH,
                                                                        num_epochs = NUM_EPOCHS,
                                                                        trial_period = TRIAL_PERIOD,
                                                                        sample_window = RMS_SAMPLING_WINDOW,
                                                                        window_stepsize = RMS_WINDOW_STEPSIZE)

    EMG, EMG_epoch, EMG_epoch_mean = EMG_pre_ins.preprocessing_routine(raw_emg = EMG_raw,
                                                                    bandpass_lowcut = EMG_LOWCUT,
                                                                    bandpass_highcut = EMG_HIGHCUT,
                                                                    num_channels = EMG_NUM_CH,
                                                                    num_epochs = NUM_EPOCHS,
                                                                    trial_period = TRIAL_PERIOD)

    EEG, EEG_epoch, EEG_epoch_mean = EEG_pre_ins.preprocessing_routine(raw_eeg = EEG_raw,
                                                                    bandpass_lowcut = EEG_LOWCUT,
                                                                    bandpass_highcut = EEG_HIGHCUT,
                                                                    num_channels = EEG_NUM_CH,
                                                                    num_epochs = NUM_EPOCHS,
                                                                    trial_period = TRIAL_PERIOD)
    
    # Release resources
    EMG_raw = None
    EEG_raw = None

    #--------------------------------------#
    # Select how data should be visualized #
    #--------------------------------------#
    #vis_EMG_ins = visualize_EMG(fs = EMG_FREQ, rms_sampling_window = RMS_SAMPLING_WINDOW, rms_windows_stepsize = RMS_WINDOW_STEPSIZE, which_finger = which_finger, num_epochs = NUM_EPOCHS, trial_period = TRIAL_PERIOD)
    vis_EEG_ins = visualize_EEG(fs = EEG_FREQ, which_finger = which_finger, num_epochs = NUM_EPOCHS, trial_period = TRIAL_PERIOD)

    #vis_EMG_ins.plot_rms_across_channels(emg = EMG_epoch_mean[which_finger], rms = RMS_epoch_mean[which_finger], markers = markers[which_finger], display_window = 0)

    vis_EEG_ins.plot_egg_across_channels(eeg = EEG[which_finger], markers = markers[which_finger], display_window = 0)

if __name__ == '__main__':
    main()