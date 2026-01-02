import time
#import argparse
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter
import os
from playsound3 import playsound
import csv
import numpy as np

from datetime import datetime # Only used for experiments


'''
q_EEG : Queue to send EEG data to main process
q_ICOM_EEG : Queue to receive instructions from main process
q_RCOM_EEG : Queue to send responses back to main process
barrier : Barrier for synchronization with other processes
'''

class EEG_con():
    def __init__(self, serial_port='USB9'):        # '/dev/ttyUSB1' linux or USB_ windows
        # BrainFlow setup
        params = BrainFlowInputParams()
        params.serial_port = serial_port  # BEWARE OF SERIAL_PORT
        board_id = BoardIds.CYTON_DAISY_BOARD
        BoardShim.enable_dev_board_logger()
        self.board = BoardShim(board_id, params)

        print("EEG - [OK] EEG collector initialized:")
        print(f"EEG - [OK] Cyton_daisy_board sampling rate: {BoardShim.get_sampling_rate(self.board.get_board_id())}")

    def insert_marker(self, board, code):
        board.insert_marker(code)

    def create_file_header(self, filepath):
        # if file doesn't exist, write header

        # MISSING
        sensor_headers = [f'ch{i}' for i in range(self.ss[0], self.ss[1] + 1)]

        if os.path.exists(filepath):
            raise FileExistsError(f"File already exists: {filepath}")


        with open(filepath, 'w', newline='') as f:
            np.savetxt(f, np.array([sensor_headers]),
                    delimiter=',', fmt='%s')

    def start(self, q_EEG, q_ICOM_EEG, q_RCOM_EEG, barrier):
        '''
        Calling this method listens for queue instructions and acts accordingly.
        Instructions:
            - Is a tuple of (command, filepath) - Index 0 is command, Index 1 is filepath
            - command: "record" or "stop"
            - filepath: path to save the recorded data (only for "record" command)
        '''
        
        self.barrier = barrier                                                              # Barrier for synchronization with other processes
        self.board.prepare_session()

        while True: # Figure out a better way to do this
            
            if not q_ICOM_EEG.empty():      # Enter if there is a queue
                
                instruction = q_ICOM_EEG.get()  # Get the instruction from the queue
                match instruction[0]:
                    case "record":
                        filepath_EEG = instruction[1]
                        self.board.start_stream()
                        t_flush = 0
                        while t_flush < 5:
                            time.sleep(1)
                            t_flush += 1
                        
                        print('EEG - [OK] Waiting for barrier')
                        self.barrier.wait()

                        self.board.get_board_data()     # Flush ring buffer
                        print(f'EEG - real time: {time.perf_counter_ns() / 1e9}')
                        
                        
                    case "stop":
                        print('EEG - [OK] Stopping EEG recording')
                        data = self.board.get_board_data()
                        self.board.stop_stream()
                        self.board.release_session()

                        np.savetxt(filepath_EEG, data.T, delimiter=',', fmt='%.6f')
                        #record_flag = False
                        break
        
        
    ''' 
    
    def execute_protocol(self):
        
        #Executes the EEG protocol by sending markers at specific intervals.
        
        
        self.board.start_stream()               

        for t in range(TRIM_TIME):              # Give some time for the stream to stabilize
            print("Session begins in", 5-t)
            time.sleep(1)                       # Wait for the stream to stabilize
        self.barrier.wait()                     # Synchronize with other processes
        t_trim = time.perf_counter_ns()
        self.insert_marker(self.board, TRIM)    # Send trim marker to BrainFlow 
        self.log_marker(t_trim, TRIM, "Trim")   # Log trim marker and time to file

        try:
            for i in range(self.num_trails):
                t0 = time.perf_counter_ns()

                # 0–1 s: Cue
                playsound(self.cue_beep, block=False)
                self.insert_marker(self.board, CUE)                 # Send marker to BrainFlow
                self.log_marker(t0, CUE, "Cue")                  # Log marker and time to file
                t_abs = self.at(t0, T_CUE)                          # Calculate absolute time for next event
                self.wait_until(t_abs)                              # Wait until the absolute time is over
                
                # 3 s: Contract
                playsound(self.contract_beep, block=False)
                self.insert_marker(self.board, CONTRACT)
                self.log_marker(t_abs, CONTRACT, "Contract")
                t_abs = self.at(t0, T_CON)
                self.wait_until(t_abs)

                # 3 s: Release
                playsound(self.release_beep, block=False)
                self.insert_marker(self.board, RELEASE)
                self.log_marker(t_abs, RELEASE, "Release")
                t_abs = self.at(t0, T_RE)
                self.wait_until(t_abs)

                # 2 s: Trial end
                self.insert_marker(self.board, TRIAL_END)
                self.log_marker(t_abs, TRIAL_END, "Trial End")
                t_abs = self.at(t0, T_END)
                self.wait_until(t_abs)

                print(f"Trial {i+1}/{self.num_trails}")

                if self.stop_event:
                    break
        
        finally:
            # Set trim marker at the end of the session and record additional 5 seconds of data
            t_trim = time.perf_counter_ns()
            self.insert_marker(self.board, TRIM)    # Send trim marker to BrainFlow 
            self.log_marker(t_trim, TRIM, "Trim")   # Log trim marker and time to file
            for t in range(TRIM_TIME):              # Give some time for the stream to stabilize
                print("Session ends in", 5-t)
                time.sleep(1)                       # Wait for the stream to stabilize
            
            data = self.board.get_board_data()
            self.board.stop_stream()
            self.board.release_session()
            print("EEG session finished")
            return data    
    

    def pure_recording(self, total_record_period, filepath_markers, filepath_EEG):
        
        #Record EEG data and write to file
        #args:
        #    total_record_period: Amount of recording in secounds
         #   filepath_markers: Path to save markers
         #   filepath_EEG: Path to save EEG data
        

        self.board.prepare_session()

        self.marker_file = open(filepath_markers, "w", newline="")
        self.writer = csv.writer(self.marker_file)
        self.writer.writerow(["timestamp_ns", "marker_id", "description"])  # header
        
        self.board.start_stream()               

        for t in range(TRIM_TIME):              # Give some time for the stream to stabilize
            print("Session begins in", 5-t)
            time.sleep(1)                       # Wait for the stream to stabilize
        t_trim = time.perf_counter_ns()
        self.insert_marker(self.board, TRIM)    # Send trim marker to BrainFlow 
        self.log_marker(t_trim, TRIM, "Trim")   # Log trim marker and time to file

        
        time.sleep(total_record_period)
        
        
        # Set trim marker at the end of the session and record additional 5 seconds of data
        t_trim = time.perf_counter_ns()
        self.insert_marker(self.board, TRIM)    # Send trim marker to BrainFlow 
        self.log_marker(t_trim, TRIM, "Trim")   # Log trim marker and time to file
        for t in range(TRIM_TIME):              # Give some time for the stream to stabilize
            print("Session ends in", 5-t)
            time.sleep(1)                       # Wait for the stream to stabilize
        
        data = self.board.get_board_data()
        self.board.stop_stream()
        self.board.release_session()
        print("EEG session finished")

        self.close_writer()
        DataFilter.write_file(data, filepath_EEG, 'w')
    
    def sound_recording(self, filepath_markers, filepath_EEG, num_trials):
        
        #Executes the EEG protocol by sending markers at specific intervals.
        

        self.board.prepare_session()

        self.marker_file = open(filepath_markers, "w", newline="")
        self.writer = csv.writer(self.marker_file)
        self.writer.writerow(["timestamp_ns", "marker_id", "description"])  # header
        
        self.board.start_stream()               

        for t in range(TRIM_TIME):              # Give some time for the stream to stabilize
            print("Session begins in", 5-t)
            time.sleep(1)                       # Wait for the stream to stabilize
        t_trim = time.perf_counter_ns()
        self.insert_marker(self.board, TRIM)    # Send trim marker to BrainFlow 
        self.log_marker(t_trim, TRIM, "Trim")   # Log trim marker and time to file

        try:
            for i in range(num_trials):
                t0 = time.perf_counter_ns()

                # 0–1 s: Cue
                playsound(self.cue_beep, block=False)
                self.insert_marker(self.board, CUE)                 # Send marker to BrainFlow
                self.log_marker(t0, CUE, "Cue")                  # Log marker and time to file
                t_abs = self.at(t0, T_CUE)                          # Calculate absolute time for next event
                self.wait_until(t_abs)                              # Wait until the absolute time is over
                
                # 3 s: Contract
                playsound(self.contract_beep, block=False)
                self.insert_marker(self.board, CONTRACT)
                self.log_marker(t_abs, CONTRACT, "Contract")
                t_abs = self.at(t0, T_CON)
                self.wait_until(t_abs)

                # 3 s: Release
                playsound(self.release_beep, block=False)
                self.insert_marker(self.board, RELEASE)
                self.log_marker(t_abs, RELEASE, "Release")
                t_abs = self.at(t0, T_RE)
                self.wait_until(t_abs)

                # 2 s: Trial end
                self.insert_marker(self.board, TRIAL_END)
                self.log_marker(t_abs, TRIAL_END, "Trial End")
                t_abs = self.at(t0, T_END)
                self.wait_until(t_abs)

                print(f"Trial {i+1}/{num_trials}")

        finally:
            # Set trim marker at the end of the session and record additional 5 seconds of data
            t_trim = time.perf_counter_ns()
            self.insert_marker(self.board, TRIM)    # Send trim marker to BrainFlow 
            self.log_marker(t_trim, TRIM, "Trim")   # Log trim marker and time to file
            for t in range(TRIM_TIME):              # Give some time for the stream to stabilize
                print("Session ends in", 5-t)
                time.sleep(1)                       # Wait for the stream to stabilize
            
            data = self.board.get_board_data()
            self.board.stop_stream()
            self.board.release_session()
            print("EEG session finished")
            
            self.close_writer()
            DataFilter.write_file(data, filepath_EEG, 'w')
'''
if __name__ == "__main__":
    EEG_ins = EEG_con()

    ###
    ### Record pure EEG
    ###
    '''
    current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    SUBJECT_NAME = "Mathias/pure_recording/"                                           # Define the subject name                               
    BASE_PATH = "/home/nicklas/Documents/master_thesis/workspace/src/dynamixel_control/recordings/"          # Define the path where the folder will be created
    print(BASE_PATH)
    filepath_EEG = BASE_PATH + SUBJECT_NAME  + "EEG/" + "EEG - " + str(current_time) + ".csv"
    filepath_markers = BASE_PATH + SUBJECT_NAME  + "Markers - " + str(current_time) + ".csv"

    directory = BASE_PATH + SUBJECT_NAME + "EEG"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    EEG_ins.pure_recording(
        total_record_period=100,    # in sec
        filepath_markers=filepath_markers,
        filepath_EEG=filepath_EEG
    )
    '''
    ###
    ### Record EEG with sounds
    ###
    current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    #SUBJECT_NAME = "Mathias/sound_recording/"                                           # Define the subject name                               
    SUBJECT_NAME = "Mathias/move_bottle/"
    BASE_PATH = "/home/nicklas/Documents/master_thesis/workspace/src/dynamixel_control/recordings/"          # Define the path where the folder will be created

    filepath_EEG = BASE_PATH + SUBJECT_NAME  + "EEG/" + "EEG - " + str(current_time) + ".csv"
    filepath_markers = BASE_PATH + SUBJECT_NAME  + "Markers - " + str(current_time) + ".csv"

    directory = BASE_PATH + SUBJECT_NAME + "EEG"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    EEG_ins.sound_recording(
        filepath_markers=filepath_markers,
        filepath_EEG=filepath_EEG,
        num_trials=10
    )


''' Sliding window by chat
# --- Initialize ---
params = BrainFlowInputParams()
board = BoardShim(BoardIds.CYTON_BOARD, params)
board.prepare_session()
board.start_stream(45000)  # internal buffer size
print("Streaming...")

fs = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD)

window_length = 1  # seconds
step_size = 0.25   # seconds

n_window = int(window_length * fs)
n_step = int(step_size * fs)

try:
    while True:
        # Get the latest n_window samples
        data = board.get_current_board_data(n_window)
        eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD)
        eeg_data = data[eeg_channels, :]

        # Do processing (e.g., filter, power, etc.)
        mean_power = np.mean(np.square(eeg_data), axis=1)
        print(f"Window mean power: {mean_power}")

        # Slide window every step_size seconds
        time.sleep(step_size)

except KeyboardInterrupt:
    print("Stopping stream...")

board.stop_stream()
board.release_session()
'''
    