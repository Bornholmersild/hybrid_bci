# Manage datasets
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from math import ceil

# Manage file paths
from pathlib import Path
import os

# Model
import torch

# Syncronization
from time import time, perf_counter, sleep

# External libraries
from src.utilities.pytrigno import TrignoEMG

# Own implementations
from src.utilities.load_and_visualize_data import load_datasets
from src.utilities.preprocessing import RejectBadEpochs
from src.models.classification_pipeline import SingleNet_CNN_LSTM_ATTENTION


# Data types
from typing import Tuple, Dict, List

from scipy.signal import iirnotch, lfilter, lfilter_zi, butter
from scipy.ndimage import median_filter
from scipy.ndimage import uniform_filter1d

#==================#
# Global variables #
#==================#
EMG_FREQ = 2000
EEG_FREQ = 125
RMS_FREQ = 40                   # 40 for 500 samples, 125 for 32 samples (window)

EEG_USEABLE_CHANNELS = [2, 3, 6, 7, 8, 9, 10, 11]

EMG_LOWCUT = 20
EMG_HIGHCUT = 450
EEG_LOWCUT = 0.5
EEG_HIGHCUT = 30

EEG_NUM_CH = len(EEG_USEABLE_CHANNELS)
EMG_NUM_CH = 3

RMS_SAMPLING_WINDOW = 200           # 500 samples - 250 ms                      32 samples - 16 ms                                       
RMS_WINDOW_STEPSIZE = 25            # 50 samples - 25 ms (90 % overlap)         16 samples - 8 ms (50 % overlap)

HAMPEL_WINDOWSIZE = 100
HAMPEL_SIGMA = 3                    # Usually 2

EMG_SELECT_SENSORS = (0, 1)
EMG_SAMPLES_PER_READ = 200

EMG_CONFIG_DICT = {
    'rms_windowsize' : RMS_SAMPLING_WINDOW,
    'rms_stepsize' : RMS_WINDOW_STEPSIZE,
    'hampel_windowsize' : HAMPEL_WINDOWSIZE,
    'hampel_sigma' : HAMPEL_SIGMA,
    'hampel_plot_option' : [False, None],
    'include_EMG' : False
}

REJECT_CONFIG_DICT = {
    'EEG_epoch_rejection_tolerance' : 6,
    'EMG_epoch_rejection_tolerance' : 6,
    'EEG_ch_acceptance' : 0,
    'EMG_ch_acceptance' : 0
}

# Routine:
# 1. Load real-time data in circular queues
# 2. Sliding window to extract data
# 3. Preprocess data
# 4. Load into model

class BandpassFilter:
    def __init__(self, fs, lowcut, highcut, order=4):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

        nyq = fs / 2
        low = lowcut / nyq
        high = highcut / nyq

        self.b, self.a = butter(order, [low, high], btype='band')
        self.zi = None

    def update(self, x):
        """
        x: shape (N, channels)
        """
        if x.ndim == 1:
            x = x[:, None]

        if self.zi is None:
            self.zi = np.tile(lfilter_zi(self.b, self.a), (x.shape[1], 1)).T

        y, self.zi = lfilter(self.b, self.a, x, axis=0, zi=self.zi)
        return y

class NotchFilter:
    def __init__(self, fs, cutoff=50, Q=30):
        self.fs = fs
        self.cutoff = cutoff
        self.Q = Q

        # Design filter
        w0 = cutoff / (fs / 2)
        self.b, self.a = iirnotch(w0, Q)

        self.zi = None  # filter state

    def update(self, x):
        """
        x: shape (N, channels)
        """
        if x.ndim == 1:
            x = x[:, None]

        if self.zi is None:
            # initialize per channel
            self.zi = np.tile(lfilter_zi(self.b, self.a), (x.shape[1], 1)).T

        y, self.zi = lfilter(self.b, self.a, x, axis=0, zi=self.zi)
        return y
    
class EMANormalizer:
    def __init__(self, alpha=0.999, eps=1e-8):
        self.alpha = alpha
        self.eps = eps

        self.mu = None
        self.var = None

    def update(self, x):
        """
        x: shape (N, channels)
        returns normalized x
        """
        x = np.asarray(x, dtype=np.float64)

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)

        if self.mu is None or self.var is None:
            self.mu = batch_mean
            self.var = batch_var

        x_norm = (x - self.mu) / (np.sqrt(self.var) + self.eps)

        # Update EMA estimates
        self.mu = self.alpha * self.mu + (1 - self.alpha) * batch_mean
        self.var = self.alpha * self.var + (1 - self.alpha) * batch_var

        return x_norm
    
    def train_EMA_coefficients(self, data : np.ndarray, sliding_window_size : int = 1000):
        if self.mu is not None or self.var is not None:
            raise ValueError("EMA coefficients have already been trained. Re-instantiate EMGStreamProcessor to reset.")
        # Train EMA coefficients by running through the data once
        for trial_start in range(0, data.shape[0], sliding_window_size):
            trial_end = trial_start + sliding_window_size
            segment = data[trial_start:trial_end]
            self.update(segment)

        print('EMA : Initilized mean:', self.mu)
        print('EMA : Initilized variance:', self.var)

class EMGStreamProcessor:
    def __init__(self, fs, lowcut, highcut,
                 reject_config_dict : Dict,
                 rms_window=200, rms_step=50,
                 hampel_window=200, hampel_sigma=3,
                 base_dir : str = 'to_data_folder'):

        self.fs = fs
        self.notch_ins = NotchFilter(fs = fs, cutoff = 50, Q = 30)
        self.bandpass_ins = BandpassFilter(fs = fs, lowcut = lowcut, highcut = highcut, order = 4)
        self.ema_ins = EMANormalizer(alpha=0.999, eps=1e-8)

        self.rms_window = rms_window
        self.rms_step = rms_step
        self.hampel_window = hampel_window
        self.hampel_sigma = hampel_sigma
        self.first_update = False
        
        self.base_dir = base_dir
        self.reject_config_dict = reject_config_dict
    
    def reset(self):
        self.notch_ins = NotchFilter(fs = self.fs, cutoff = 50, Q = 30)
        self.bandpass_ins = BandpassFilter(fs = self.fs, lowcut = EMG_LOWCUT, highcut = EMG_HIGHCUT, order = 4)
        self.ema_ins = EMANormalizer(alpha=0.999, eps=1e-8)
        self.first_update = False

    def update(self, chunk):
        """
        Preprocess a chunk of EMG data through the pipeline:
        1) Notch filter (50 Hz)
        2) Bandpass filter (20-450 Hz)
        3) Hampel filter (window size = 100, n_sigmas = 2.0)
        4) RMS (window size = 200, step size = 50)
        5) EMA normalization
        """

        #===========================#
        # 1) Filtering (continuous) #
        #===========================#
        notch = self.notch_ins.update(chunk)
        bandpass = self.bandpass_ins.update(notch)

        if not self.first_update:               # Don't append first update to avoid filter transients
            self.first_update = True
            return None

        #==================#
        # 2) Hampel filter #
        #==================#
        hampel = self._hampel(bandpass, window_size = self.hampel_window, n_sigmas = self.hampel_sigma)

        #==================#
        # 3) RMS (sliding) #
        #==================#
        rms = self._rms_conv(hampel, window_size = self.rms_window, step_size = self.rms_step)

        #==================#
        # 4) Normalization #
        #==================#
        # norm = self.ema_ins.update(rms)

        return rms

    def load_subject_data(self, subj : str, finger : str, modality : str, trim_period : int = 3, trial_period : int = 9):
        '''
        This method is only used for loading data for real-time classification.
        It loads, trims the edges and move bad epochs
        '''
        load_ins = load_datasets(base_dir = self.base_dir)
        reject_ins = RejectBadEpochs(base_dir = self.base_dir)

        file_path = load_ins.find_flex_files(subjects = subj,
                                             modality = modality,
                                             fingers = finger,
                                             prefix = 'flex'
                                             )
        
        data_container = []
        epochs_overview = []

        for file in file_path:
            print(file)
            data = pd.read_csv(file).to_numpy()

            #============#
            # Trim edges #
            #============#
            trim_samples = self.fs * trim_period
            samples_per_epoch = self.fs * trial_period

            valid_samples = data.shape[0] - 2 * trim_samples            # Total samples for experimental period. WHY *2 : Trim egde on both sides
            num_epochs = int( np.round(valid_samples / samples_per_epoch) )     # Divide out total samples in sections of samples per epoch -> Results in number of epochs

            trim_start = trim_samples
            trim_end = trim_start + num_epochs * samples_per_epoch              # WHY instead of data[trim : -trim] -> Inconsistency in protocol causes the last batch of data not be included -> Rare but can happen

            data_trim = data[trim_start : trim_end, :]

            data_container.append(data_trim)
            epochs_overview.append(num_epochs)
            
        data_container = np.concatenate(data_container, axis=0)

        #===================#
        # Reject bad epochs #
        #===================#
        reject_mask = reject_ins.reject_routine(data_file_per_finger = file_path,
                                            epochs_overview = epochs_overview,
                                            EEG_data = None,
                                            RMS_data = data_container,
                                            reject_config_dict = self.reject_config_dict,
                                            EEG_useable_channels = None)

        total_epochs = sum(epochs_overview)
        EMG_epoch = data_container.reshape(total_epochs, data_container.shape[0] // total_epochs, data_container.shape[1])

        EMG_epoch_clean = EMG_epoch[~reject_mask]

        total_clean_epochs = EMG_epoch_clean.shape[0]

        return EMG_epoch_clean, total_clean_epochs
    
    def relabel_windows(self, epochs : np.ndarray, window_samples : int = 1000, step_samples : int = 200, fs : int = EMG_FREQ, labels : list = ["rest", "contract", "release"]):
        '''
        This method takes the epochs and converts to continuous data and relabels it according to the experimental protocol.
        It produces a window of 1000 samples and a step size of 200 samples.
        Each period will contain 26 windows. 1 trial = 78 windows

        Returns
        -------
        filtered_epochs : np.ndarray
            Shape (n_windows, n_samples, n_channels)
            Each window is trial*periods*steps (26 steps per period)
        '''
        WINDOW_SAMPLES = window_samples
        STEP_SAMPLES = step_samples

        TRIAL_SAMPLES = 9 * fs
        SEGMENT_SAMPLES = 3 * fs

        labels = labels

        window_labels = []
        filtered_epochs = []

        data = epochs.reshape(-1, epochs.shape[-1])
        n_samples = data.shape[0]

        self.reset()        # For every subject reset the filters and EMA normalizer to avoid data leakage between subjects

        # Loop over trials in continuous data
        for trial_start in range(0, n_samples, TRIAL_SAMPLES):

            trial_end = trial_start + TRIAL_SAMPLES
            if trial_end > n_samples:
                break

            # Loop over segments inside trial
            for seg_idx in range(3):
                seg_start = trial_start + seg_idx * SEGMENT_SAMPLES
                seg_end   = seg_start + SEGMENT_SAMPLES

                segment = data[seg_start:seg_end]

                # Sliding window inside segment
                for start in range(0, SEGMENT_SAMPLES - WINDOW_SAMPLES + 1, STEP_SAMPLES):
                    end = start + WINDOW_SAMPLES

                    chunk = segment[start:end]

                    filtered_data = self.update(chunk)

                    if filtered_data is None:
                        continue
                    
                    if seg_idx == 1:
                        pass
                    # Ensure 1sec of contract and the last 2 sec of release are labeled rest
                    if seg_idx == 1 and end <= fs:     # Until 1 sec of contract will be labeled rest
                        label = "rest"
                    elif seg_idx == 2 and start > fs:  # Beyond 1 sec of release will be labeled rest
                        label = "rest"
                    else:
                        label = labels[seg_idx]

                    window_labels.append(label)
                    filtered_epochs.append(filtered_data)
        
        return np.array(filtered_epochs), np.array(window_labels)

    def _hampel(self, x: np.ndarray, window_size: int = 100, n_sigmas: float = 3.0, plot_filter_results: list = [False, None]):
        """
        Hampel filter for multi-channel signals.

        Parameters
        ----------
        x : np.ndarray
            Shape (n_samples, n_channels)
        window_size : int
            Number of samples on EACH side of the center sample
        n_sigmas : float
            Threshold multiplier
        plot_filter_results : list
            First element -> bool to display filter results
            second element -> list of specfic time window or None to display all

        Returns
        -------
        filtered_data   : np.ndarray (n_samples, n_channels)
        """

        x = np.asarray(x, dtype=float)

        if x.ndim != 2:
            raise ValueError("Input must be 2D: (samples, channels)")

        # Scale factor to make MAD comparable to standard deviation
        # (valid for approximately Gaussian data)
        k_scale = 1.4826
        kernel = 2 * window_size + 1

        # Median per channel (filter only along time axis)
        medians = median_filter(
            x,
            size=(kernel, 1),
            mode="reflect"
        )

        # Difference between real signal and the typical values of the signal
        diff = np.abs(x - medians)

        # Robust estimate of local variability (Median Absolute Deviation) per channel
        mad = k_scale * median_filter(
            diff,
            size=(kernel, 1),
            mode="reflect"
        )

        thresholds = n_sigmas * mad

        outlier_mask = diff > thresholds

        x_filt = x.copy()
        x_filt[outlier_mask] = medians[outlier_mask]

        if plot_filter_results[0]:
            # Collect outlier indices per channel
            outlier_indices = [
                np.nonzero(outlier_mask[:, ch])[0].tolist()
                for ch in range(x.shape[1])
            ]
            for ch in range(x.shape[1]):
                print('Number of outliers:', len(outlier_indices[ch])) 

            self.plot_hampel_filter(original_signal = x, filtered_signal = x_filt, outlier_indices = outlier_indices, medians = medians, thresholds = thresholds, zoom = plot_filter_results[1])
        
        return x_filt

    def _rms_conv(self, signal, window_size=200, step_size=25):
        '''
        Convolution RMS. RMS = sqrt( LPF(x^2) ), where LPF is implemented as a uniform filter (moving average) over the squared signal.
        '''
        power = signal**2

        mean_power = uniform_filter1d(
            power,
            size=window_size,
            axis=0,
            mode="nearest"
        )

        rms = np.sqrt(mean_power)

        return rms[::step_size]
  
class EMGRealTime:
    def __init__(self, config_dict : Dict, select_sensors : Tuple = (0, 2), samples_per_read : int = 200, units : str = 'mV'):
        self.config = config_dict
        self.EMG_ins = TrignoEMG(channel_range = select_sensors, samples_per_read = samples_per_read, units = units)

    def start_stream(self):
        print('EMG stream initilizing')
        self.EMG_ins.start()
        t0 = time()
        current_read = 0
 
        while (current_read < 1):
            current_read = np.mean(self.EMG_ins.read())

            tnow = time()
            if tnow > t0 + 7:
                raise TimeoutError('Timeout for EMG initilization')
            
        print('EMG ready to GO!')

    def end_stream(self):
        print('EMG stream terminated')
        self.EMG_ins.stop()

    def extract_data(self):
        return self.EMG_ins.read()

class Buffer:
    def __init__(self, max_size : int, num_ch : int, window_size : int, step_size : int):
        # Parameters for circular buffer
        self.max_size = max_size
        self.num_ch = num_ch
        self.buffer = np.zeros((max_size, num_ch))
        self.current_size = 0
        self.write_idx = 0

        # Parameters for sliding window
        self.window_size = window_size
        self.step_size = step_size              # Step_size equal to window_size means no overlap, step_size < window_size means overlap, step_size > window_size means gap between windows
        self.read_idx = 0                       # Track pointer

    def add_data(self, data : np.ndarray):
        n_samples = data.shape[0]

        # Last position where there is new data
        end_idx = (self.write_idx + n_samples) % self.max_size

        if end_idx < self.write_idx:                                # Overwrite old data (wrap around)
            split = self.max_size - self.write_idx                  # Number of samples that can be written to end of buffer before wrapping around
            self.buffer[self.write_idx:] = data[:split]             # Write first part of new data to end of buffer
            self.buffer[:end_idx] = data[split:]                    # Write remaining new data to beginning of buffer
        else:                                                       # Update data (No wrap around needed)
            self.buffer[self.write_idx:end_idx] = data              # Write new data into buffer
        
        self.write_idx = end_idx
        self.current_size = min(self.current_size + n_samples, self.max_size)   # Update current size of buffer (cannot exceed max size)

    def get_window(self):
        if self.current_size < self.window_size:
            print('Not enough data in buffer to extract window')
            return None
        
        start = self.read_idx
        end = start + self.window_size

        if end <= self.max_size:
            window = self.buffer[start:end]                         # Extract window of data from buffer (no wrap around)
        else:
            window = np.vstack((self.buffer[start:], 
                                self.buffer[:end % self.max_size])) # Extract window of data from buffer (wrap around)    

        # Update read pointer
        self.read_idx = (self.read_idx + self.step_size) % self.max_size

        return window

class Model():
    def __init__(self, path_to_model : Path):
        self.path_dir = path_to_model
        self.model = None
        self.device = None

        self.initilize_model()

        self.pred_mapping = {
            0 : 'Index Contract',
            1 : 'Index Release',
            2 : 'Thumb Contract',
            3 : 'Thumb Release',
            4 : 'Rest'
        }

    def initilize_model(self):
        if not os.path.exists(self.path_dir):
            raise FileExistsError(self.path_dir)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #pin_memory = torch.cuda.is_available()

        #======================#
        # Load model arguments #
        #======================#
        checkpoint = torch.load(f = self.path_dir / "model.pth", map_location = self.device)
        model_args = checkpoint["model_args"]

        self.model = SingleNet_CNN_LSTM_ATTENTION(**model_args)

        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)

        self.model.eval()

        print('Model initilized')
    
    def predict(self, input_data):
        inp = torch.tensor(input_data, dtype = torch.float32).to(self.device).unsqueeze(0)          # Match model dim (1, S, Ch)

        with torch.no_grad():
            logits, _, _ = self.model(inp)
            pred_idx = torch.argmax(logits, dim=1).item()
        
        pred_map = self.pred_mapping[pred_idx]

        return pred_map
        
def examine_latency():
    base_dir = Path(__file__).resolve().parent / 'data'
    print(base_dir)
    find_files_ins = load_datasets(base_dir = base_dir)
    find_file = find_files_ins.find_flex_files
    subjects = [f'subject_{i}' for i in range(0, 2)]

    DATA = []
    for subj in subjects:
        data_path = find_file(subjects = subj,
                                modality = 'EMG',
                                fingers = 'index',
                                prefix = 'flex')
        
        subj_data = []
        
        for file in data_path:
            data = pd.read_csv(file).to_numpy()
            
            subj_data.append(data)
        
        DATA.append(np.concatenate(subj_data, axis = 0))
        
    
    DATA = np.concatenate(DATA, axis = 0)
    print(DATA.shape)

    EMG = EMGRealTime(config_dict = EMG_CONFIG_DICT,
                      select_sensors = EMG_SELECT_SENSORS,
                      samples_per_read = EMG_SAMPLES_PER_READ)
    
    EMG_BUFFER = Buffer(max_size = 2000,
                        num_ch = EMG_NUM_CH,
                        window_size = RMS_SAMPLING_WINDOW,
                        step_size = RMS_WINDOW_STEPSIZE)
    
    t_buffer = []
    t_preprocess = []

    for chunk in range(0, DATA.shape[0], 500):
        
        chunk_data = DATA[chunk:chunk+500]          # Read data
        t0 = perf_counter()

        EMG_BUFFER.add_data(chunk_data)             # Load into circular buffer

        data_window = EMG_BUFFER.get_window()       # Extract window of data by sliding window

        t_buffer_temp = perf_counter()

        data_clean = EMG.preprocess(data_window)                    # Preprocess window of data

        t_preprocess_temp = perf_counter()

        t_buffer.append(t_buffer_temp - t0)
        t_preprocess.append(t_preprocess_temp - t_buffer_temp)
    
    print(f'Average time for buffer operations: {np.mean(t_buffer) * 1000:.2f} ms')
    print(f'Average time for preprocessing: {np.mean(t_preprocess) * 1000:.2f} ms')

def Trigno_test():
    EMG = EMGRealTime(config_dict = EMG_CONFIG_DICT,
                    select_sensors = (0, 1),
                    samples_per_read = EMG_SAMPLES_PER_READ)
    
    EMG.start_stream()

    print('Sleep')
    tim = 3
    sleep(tim)

    EMG.end_stream()

    data = EMG.extract_data()

    for i in range(5*tim + 5):
        print(i)
        data = np.array(data)
        print(data.shape)
        print(data.mean())
    

def main(model_folder_name : 'str' = 'SingleNet_CNN+LSTM+ATTENTION_EMG/subject_0'):
    model_path_folder = Path(__file__).resolve().parents[2] / f"models/loggings/real_time/{model_folder_name}"

    STREAM = EMGRealTime(config_dict = EMG_CONFIG_DICT,
                      select_sensors = EMG_SELECT_SENSORS,
                      samples_per_read = EMG_SAMPLES_PER_READ)
    
    EMG_BUFFER = Buffer(max_size = 10000,
                        num_ch = EMG_NUM_CH,
                        window_size = RMS_SAMPLING_WINDOW,
                        step_size = RMS_WINDOW_STEPSIZE)
    
    PREPROCESS = EMGStreamProcessor(fs = EMG_FREQ, lowcut = EMG_LOWCUT, highcut = EMG_HIGHCUT,
                                    reject_config_dict = EMG_CONFIG_DICT, 
                                    rms_window = RMS_SAMPLING_WINDOW, rms_step = RMS_WINDOW_STEPSIZE,
                                    hampel_window = HAMPEL_WINDOWSIZE, hampel_sigma = HAMPEL_SIGMA,     # sigma usually 2
                                    base_dir = 'Unused')
    
    MODEL = Model(path_to_model = model_path_folder)

    
    STREAM.start_stream()                      # Initilize streaming
    '''
    #=================================================================#
    num_channels = EMG_SELECT_SENSORS[1] - EMG_SELECT_SENSORS[0] + 1
    FS = 2000
    WINDOW_SEC = 1
    BUFFER_LEN = FS * WINDOW_SEC

    plt.ion()
    n_cols = 2
    n_rows = ceil(num_channels / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6), sharex=True)
    axes = axes.flatten()

    x = np.arange(BUFFER_LEN) / FS
    lines = []

    for ch in range(num_channels):
        ax = axes[ch]
        line, = ax.plot(x, np.zeros(BUFFER_LEN), lw=1)
        ax.set_title(f"Sensor {ch}")
        ax.set_ylim(-0.1, 0.1)           # adjust after seeing real envelope ranges
        ax.set_xlim(0, WINDOW_SEC)
        lines.append(line)

    # Hide unused axes
    for i in range(num_channels, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

                # Update plot lines
            for ch in range(num_channels):
                lines[ch].set_ydata(X_win[ch])

            plt.pause(0.001)  # allow GUI to update (very small delay)

    plt.ioff()
    plt.show()
    '''

    #=================================================================#
    time_tracker = []
    buffer_fill_size = 1
    try:
        while True:
            X_emg = STREAM.extract_data()              # Read data            

            t0 = time()

            # Load into circular buffer
            EMG_BUFFER.add_data(data = X_emg)

            if buffer_fill_size <= 5:
                buffer_fill_size += 1
                continue
            
            # Extract window of data by sliding window
            X_win = EMG_BUFFER.get_window()

            # Preprocess window of data
            X_pre = PREPROCESS.update(chunk = X_win)                # Without normalization

            # Insert into model
            X_pred = MODEL.predict(input_data = X_pre)

            # Output of the model
            print(X_pred)

            time_diff = (time() - t0) * 1000
            time_tracker.append(time_diff)


    except KeyboardInterrupt:
        print('Terminate program')
    
    finally:
        STREAM.end_stream()
        if len(time_tracker) > 0:
            print(f'Average time after extract data {np.mean(time_tracker):.2f} ms')
        

if __name__ == "__main__":
    main()
    # Trigno_test()
    # examine_latency()
    # print( ((6000 - 500) / 50) + 1)



