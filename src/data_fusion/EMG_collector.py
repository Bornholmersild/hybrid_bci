from ..utilities.pytrigno import TrignoEMG, TrignoAccel
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
        #self.imu = TrignoAccel(channel_range = self.ss, samples_per_read = self.spr)

        print(f'EMG - Number of EMG sensors {self.num_ch}')
        print("EMG - [OK] EMG collector initialized with sensors:", self.ss)

    def create_file_header(self, filepath):
        # if file doesn't exist, write header
        '''
        sensor_headers = [f"{typ} ch{i}"
                  for typ in ("EMG", "IMU")
                  for i in range(self.ss[0], self.ss[1] + 1)]
        '''
        sensor_headers = [f'ch{i}' for i in range(self.ss[0], self.ss[1] + 1)]

        if os.path.exists(filepath):
            raise FileExistsError(f"File already exists: {filepath}")

        headers = sensor_headers + ['read_time']

        with open(filepath, 'w', newline='') as f:
            #np.savetxt(f, np.array(headers), delimiter=',', fmt='%s')
            f.write(','.join(headers) + '\n')
            
    def start(self, q_EMG, q_ICOM_EMG, q_RCOM_EMG, barrier):
        
        record_flag = False
        file_handle = None
        sample_counter = 0

        while True:

            if not q_ICOM_EMG.empty():
                instruction = q_ICOM_EMG.get()
                
                match instruction[0]:
                    case 'record':
                        filepath_EMG = instruction[1]
                        # self.emg.start()
                        #self.imu.start()
                        self.create_file_header(filepath=filepath_EMG)
                        file_handle = open(filepath_EMG, 'a', buffering=1)
                        
                        # print('EMG - [WAIT] Need to flush data')
                        # t_flush = time.time()
                        # while np.all(flush_data[0, :] == 0): 
                        #     flush_data = self.emg.read().T         # Flush ring buffer
                        #     print(flush_data.shape[0])
                        #     if time.time() - t_flush > 5:           # seconds timeout
                        #         raise TimeoutError('EMG - [WARN] No signal received after 5 seconds')

                        #self.emg.read()        PUT IT BACK HERE?
                        print("EMG - [WAIT] WAITING for other processes. Dummy method.\n Begin in...")
                        for i in range(7):
                            print(f'MVC protocol starts in {7-i}', end='\r')
                            time.sleep(1)

                        self.emg.start()
                        self.emg.read()
                        print("EMG - [OK] Waiting for barrier.")
                        barrier.wait()
                        
                        #self.imu.read()
                        #flush_data = self.emg.read().T
                        #print(flush_data.shape[0])
                        record_flag = True

                    case 'stop':
                        print("EMG - [OK] Stopping EMG recording.")
                        record_flag = False
                        self.emg.stop()
                        #self.imu.stop()

                        if file_handle:
                            file_handle.close()
                            file_handle = None
                        break
            
            if record_flag:
                #read_time[-1] = (time.perf_counter_ns() - t0) / 1e9
                block = self.emg.read().T                   # shape (num_ch, samples_per_read)
                
                # Compute elapsed times for each sample
                times = (np.arange(sample_counter, sample_counter + self.spr) / self.emg.rate).reshape(-1, 1)
                sample_counter += self.spr
                data_to_save = np.hstack((block, times))
                
                np.savetxt(file_handle, data_to_save, delimiter=',', fmt='%.6f')
                

if __name__ == "__main__":
    print("Imports of EMG_collector.py successful")
    emg = EMG_con((0,2), 200, 'mV')