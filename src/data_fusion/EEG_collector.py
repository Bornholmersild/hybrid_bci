import time
#import argparse
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import os
import numpy as np


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

        if os.path.exists(filepath):
            raise FileExistsError(f"File already exists: {filepath}")

        sensor_headers = ['Sample_index', 'EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3', 'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6', 'EXG Channel 7', 'EXG Channel 8', 'EXG Channel 9', 'EXG Channel 10', 'EXG Channel 11', 'EXG Channel 12', 'EXG Channel 13', 'EXG Channel 14', 'EXG Channel 15', 'Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2', 'Not Used', 'Digital Channel 0 (D11)', 'Digital Channel 1 (D12)', 'Digital Channel 2 (D13)', 'Digital Channel 3 (D17)', 'Not Used', 'Digital Channel 4 (D18)', 'Analog Channel 0', 'Analog Channel 1', 'Analog Channel 2', 'Timestamp', 'Marker Channel', 'Timestamp (Formatted)']

        with open(filepath, 'w', newline='') as f:
            np.savetxt(f, np.array([sensor_headers]),
                    delimiter=',', fmt='%s')

    def start(self, q_EEG, q_ICOM_EEG, q_RCOM_EEG, barrier_exec):
        '''
        Calling this method listens for queue instructions and acts accordingly.
        Instructions:
            - Is a tuple of (command, filepath) - Index 0 is command, Index 1 is filepath
            - command: "record" or "stop"
            - filepath: path to save the recorded data (only for "record" command)
        '''

        self.board.prepare_session()

        while True:
            
            if not q_ICOM_EEG.empty():          # Enter if there is a queue
                instruction = q_ICOM_EEG.get()  # Get the instruction from the queue
                
                match instruction[0]:
                    case "record":
                        filepath_EEG = instruction[1]
                        self.create_file_header(filepath = filepath_EEG)

                        print('EEG - Waiting for barrier')
                        barrier_exec.wait()
                        # MAYBE MOVE START_STREAM BEFORE WAIT IN BOTH EEG AND EMG and SET A TIMER FOR PROTOCOL
                        self.board.start_stream()
                        self.board.get_board_data()     # Flush ring buffer
                        print(f'EEG - Time after flush: {time.perf_counter_ns() / 1e9}')
                        
                    case "stop":
                        print('EEG - Stopping EEG recording')
                        data = self.board.get_board_data()
                        self.board.stop_stream()
                        self.board.release_session()

                        np.savetxt(filepath_EEG, data.T, delimiter=',', fmt='%.6f')
                        break
        
if __name__ == "__main__":
   pass
    