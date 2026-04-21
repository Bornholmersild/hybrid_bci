# Manage datasets
import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
# from math import ceil

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
from src.models.classification_pipeline import SingleNet_CNN_LSTM_ATTENTION, EMGStreamProcessor


# Data types
from typing import Tuple, Dict #List

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

SLIDING_WINDOW_SAMPLES = 1000
SLIDING_WINDOW_STEPSIZE = 200

EMG_SELECT_SENSORS = (0, 2)
EMG_SAMPLES_PER_READ = 200

state = "REST"

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

class EMGRealTime:
    def __init__(self, config_dict : Dict, select_sensors : Tuple = (0, 2), samples_per_read : int = 200, units : str = 'mV'):
        self.config = config_dict
        self.EMG_ins = TrignoEMG(channel_range = select_sensors, samples_per_read = samples_per_read, units = units)

    def start_stream(self):
        print('EMG stream initilizing')
        self.EMG_ins.start()

        t0 = time()
        flush_buffer = True
 
        while flush_buffer:
            
            
            self.EMG_ins.read()
            
            tnow = time()
            if tnow > t0 + 5:
                flush_buffer = False
            
        print('EMG ready to GO!')

    def end_stream(self):
        print('EMG stream terminated')
        self.EMG_ins.stop()

    def extract_data(self):
        return np.transpose(self.EMG_ins.read())

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
        
        probs = torch.softmax(logits, dim=1)
        confidence = probs[0, pred_idx].item()

        return pred_map, confidence
    
class StateLogic():
    def __int__(self):
        pass

    def update(self, pred, confidence):
        global state

        if confidence < 0.6:
            return state

        if state == "REST":
            if pred == "Index Contract":
                state = "INDEX_ACTIVE"
            elif pred == "Thumb Contract":
                state = "THUMB_ACTIVE"

        elif state == "INDEX_ACTIVE":
            if pred == "Index Release":
                state = "REST"

        elif state == "THUMB_ACTIVE":
            if pred == "Thumb Release":
                state = "REST"

        return state

''' examine_latency
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

        # data_clean = EMG.preprocess(data_window)                    # Preprocess window of data

        t_preprocess_temp = perf_counter()

        t_buffer.append(t_buffer_temp - t0)
        t_preprocess.append(t_preprocess_temp - t_buffer_temp)
    
    print(f'Average time for buffer operations: {np.mean(t_buffer) * 1000:.2f} ms')
    print(f'Average time for preprocessing: {np.mean(t_preprocess) * 1000:.2f} ms')'''

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
    model_path_folder = Path(__file__).resolve().parents[1] / f"models/loggings/real_time/{model_folder_name}"

    MODEL = Model(path_to_model = model_path_folder)

    STREAM = EMGRealTime(config_dict = EMG_CONFIG_DICT,
                      select_sensors = EMG_SELECT_SENSORS,
                      samples_per_read = EMG_SAMPLES_PER_READ)
    
    EMG_BUFFER = Buffer(max_size = 10000,
                        num_ch = EMG_NUM_CH,
                        window_size = SLIDING_WINDOW_SAMPLES,
                        step_size = SLIDING_WINDOW_STEPSIZE)
    
    PREPROCESS = EMGStreamProcessor(fs = EMG_FREQ, lowcut = EMG_LOWCUT, highcut = EMG_HIGHCUT,
                                    reject_config_dict = EMG_CONFIG_DICT, 
                                    rms_window = RMS_SAMPLING_WINDOW, rms_step = RMS_WINDOW_STEPSIZE,
                                    hampel_window = HAMPEL_WINDOWSIZE, hampel_sigma = HAMPEL_SIGMA,     # sigma usually 2
                                    base_dir = 'Unused')
    
    STATE = StateLogic()

    # mu = np.load(model_path_folder / "mu.npy")
    # sigma = np.load(model_path_folder / "sigma.npy")

    
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
            
            if buffer_fill_size < 5:
                buffer_fill_size += 1
                continue
            
            # Extract window of data by sliding window
            X_win = EMG_BUFFER.get_window()
            
            # Preprocess window of data
            X_pre = PREPROCESS.update(chunk = X_win)                # Without normalization

            if X_pre is None:
                print('Pre is none')
                continue
            
            # Normalize
            # X_norm = (X_pre - mu) / (sigma + 1e-8)

            # Insert into model
            X_pred, confidence = MODEL.predict(input_data = X_pre)
            
            # Output of the model
            state = STATE.update(pred = X_pred, confidence = confidence)
            print(
            f"STATE: {state:<15} | "
            f"PRED: {X_pred:<20} | "
            f"CONF: {confidence:>6.2f}",
            end="\r"
            )

            time_diff = (time() - t0) * 1000
            time_tracker.append(time_diff)

            if time_diff > 200:
                print('Time different is exceed - Prediction behind')
            

    except KeyboardInterrupt:
        print('Terminate program')
    
    finally:
        STREAM.end_stream()
        if len(time_tracker) > 0:
            print(f'Average time after extract data {np.mean(time_tracker):.2f} ms')
        

if __name__ == "__main__":
    model_folder_name = 'SingleNet_CNN+LSTM+ATTENTION_EMG_complexModel_noNorm/subject_0'
    
    main(model_folder_name = model_folder_name)



