from ..utilities.pytrigno import TrignoEMG
import numpy as np
import os
import time

class EMG_con():
    def __init__(self, select_sensors = (0, 2), samples_per_read=200, units = 'mV'):
        '''
        args:
        select_sensors: tuple of int
            Indices of the selected EMG sensors (0-indexed).
        samples_per_read: int
            Number of samples to read in each read operation.
        '''
        self.spr = samples_per_read
        self.ss = select_sensors
        self.num_ch = len(self.ss)
        self.emg = TrignoEMG(channel_range = self.ss, samples_per_read = self.spr, units = units)

        print(f"EMG - Number of EMG sensors {self.num_ch}")
        print("EMG - EMG collector initialized with sensors:", self.ss)

    def create_file_header(self, filepath):
        # if file doesn't exist, write header
        if os.path.exists(filepath):
            raise FileExistsError(f"File already exists: {filepath}")

        sensor_headers = [f'ch{i}' for i in range(self.ss[0], self.ss[1] + 1)]

        with open(filepath, 'w', newline='') as f:
            #np.savetxt(f, np.array(headers), delimiter=',', fmt='%s')
            f.write(','.join(sensor_headers) + '\n')
            
    def start(self, q_EMG, q_ICOM_EMG, q_RCOM_EMG, barrier_exec, stream_on_event):
        
        record_flag = False
        file_handle = None

        while True:

            if not q_ICOM_EMG.empty():
                instruction = q_ICOM_EMG.get()
                
                match instruction[0]:
                    case 'record':
                        filepath_EMG = instruction[1]
                        self.create_file_header(filepath=filepath_EMG)
                        file_handle = open(filepath_EMG, 'a', buffering=1)

                        print("EMG - Waiting for barrier.")
                        barrier_exec.wait()
                        # MAYBE MOVE START_STREAM BEFORE WAIT IN BOTH EEG AND EMG and SET A TIMER FOR PROTOCOL
                        exit_while = False
                        self.emg.start()
                        while not stream_on_event.is_set():
                            if not exit_while:
                                q_RCOM_EMG.put('True')
                                exit_while = True
                            self.emg.read()
                            stream_on_event.wait(timeout=0.01)

                        #self.emg.read()
                        print(f'EMG - Start stream: {time.time()}')

                        record_flag = True

                    case 'stop':
                        print("EMG - Stopping EMG recording.")
                        record_flag = False
                        self.emg.stop()

                        if file_handle:
                            file_handle.close()
                            file_handle = None
                        break
            
            if record_flag:
                block = self.emg.read().T                   # shape (num_ch, samples_per_read)
                np.savetxt(file_handle, block, delimiter=',', fmt='%.6f')
                

if __name__ == "__main__":
    print("Imports of EMG_collector.py successful")
    emg = EMG_con((0,2), 200, 'mV')