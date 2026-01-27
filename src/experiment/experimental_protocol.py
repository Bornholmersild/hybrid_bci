import time
import numpy as np
from multiprocessing.synchronize import Barrier  # Import the proper Barrier type
from multiprocessing import JoinableQueue
import pygame
import os
from typing import Any
#from pydub import AudioSegment         # Used to trim audio

class PROTOCOL_con():
    """
    Protocol contructor
        
    Parameter
    ---------
    num_epochs : int 
        Number of epochs to run
    rest_duration : int 
        Duration of rest period in seconds. Defaults to 2.
    onset_duration : int 
        Duration of motion period in seconds. Defaults to 4.
    release_duration : int 
        Duration of release period in seconds. Defaults to 2.
    trim_duration : int
        Duration of collecting trach data in seconds. Default 3.
    """
    def __init__(self, 
                 num_epochs: int, 
                 rest_duration: int = 2, 
                 onset_duration: int = 4, 
                 release_duration: int = 2,
                 trim_duration : int = 3):
        
        #self.CON_SOUND, self.REL_SOUND, self.RES_SOUND = self.initialize_soundplay()

        self.REST_ID = 10
        self.ONSET_ID = 20
        self.REL_ID = 30
        self.ST_TRIM_ID = 111
        self.END_TRIM_ID = 222

        self.num_epochs = num_epochs
        self.t_rest = rest_duration
        self.t_onset = onset_duration
        self.t_rel = release_duration
        self.t_trim = trim_duration

        # GUI IMPLEMENTAION
        self.cmd_switch_text = 'switch_to_text'
        self.cmd_switch_video = 'switch_to_video'

    def start(self, 
              q_ICOM_PRO : JoinableQueue, 
              q_RCOM_PRO : JoinableQueue, 
              barrier_exec : Barrier,
              stream_on_event : Any,
              q_log_EEG : JoinableQueue,
              q_log_EMG : JoinableQueue):
        '''
        Calling this method listens for queue instructions and acts accordingly.
        '''
        execute_flag = False
        file_handle = None
        finish = False
        epoch_idx = 0
        t0 = 0

        send_queues_dict = {      
            'EEG': q_log_EEG,
            'EMG': q_log_EMG
        }

        while True:

            if not q_ICOM_PRO.empty():
                instruction = q_ICOM_PRO.get()

                match instruction[0]:
                    case 'record':
                        filepath_PRO = instruction[1]
                        self.create_file_header(filepath = filepath_PRO)
                        file_handle = open(filepath_PRO, 'a', buffering = 1, newline = '')
                        
                        for i in range(3):                      # Sets a timer in the gui before starting
                            payload = f'Begins in {3-i}'
                            cmd = self.cmd_switch_text
                            q_RCOM_PRO.put((cmd, payload))
                            time.sleep(1)
                        
                        barrier_exec.wait()                     # Can be removed when using stream_on_event?
                        print('ALL - Barriers are reached')

                        stream_on_event.wait()
                        
                        t0 = time.perf_counter_ns()
                        print(f'PROTOCOL - begin at time: {time.time()}')
                        
                        q_RCOM_PRO.put((self.cmd_switch_text, 'REST YOUR HAND'))
                        self.execute_trim_period(t0 = t0, file_handler = file_handle, send_queues_dict = send_queues_dict)
                        execute_flag = True
                        q_RCOM_PRO.put(self.cmd_switch_video)

                    case 'stop':
                        if not execute_flag:    # Break out of system, if 'record' never have been called
                            break
                        execute_flag = False    # Avoid executing the protocol 
                        finish = True           # Finish the experiment with trim period
            
            if execute_flag:
                finish = self.execute_protocol(t0 = t0, epoch_idx = epoch_idx, file_handler = file_handle, q_RCOM_PRO = q_RCOM_PRO, send_queues_dict = send_queues_dict)
                epoch_idx += 1

            if finish:
                q_RCOM_PRO.put((self.cmd_switch_text, 'GREAT JOB!'))
                self.execute_trim_period(t0 = t0, file_handler = file_handle, send_queues_dict = send_queues_dict)
                execute_flag, finish = False, False
                q_RCOM_PRO.put('terminate')         # Terminate the gui
                if file_handle:
                    file_handle.close()
                    file_handle = None
                break

    def execute_protocol(self, 
                         t0 : int,
                         epoch_idx : int,
                         file_handler,
                         q_RCOM_PRO : JoinableQueue,
                         send_queues_dict : dict):
        """Execute the experimental protocol.
        
        Args:
            num_epochs (int): Number of epochs to run
            rest_duration (float, optional): Duration of rest period in seconds. Defaults to 5.0.
            action_duration (float, optional): Duration of action period in seconds. Defaults to 5.0.
            release_duration (float, optional): Duration of release period in seconds. Defaults to 5.0.
            filepath (str, optional): Path to save markers. Defaults to None.
            barrier (multiprocessing.Barrier, optional): Synchronization barrier. Defaults to None.
        """
        q_RCOM_PRO.put('rest')
        print('REST')
        t_epoch = time.perf_counter_ns()

        # Rest period
        self.put_marker_to_queue(send_queues_dict=send_queues_dict, marker_id = self.REST_ID)
        self.log_marker(file_handler, self.diff(t0, t_epoch), marker_id = self.REST_ID, description = "Rest period started")
        t_wait = self.at(t_epoch, self.t_rest)
        self.wait_until(t_wait)

        # Execute the action
        q_RCOM_PRO.put('contract')
        print('Contract')
        self.put_marker_to_queue(send_queues_dict=send_queues_dict, marker_id = self.ONSET_ID)
        self.log_marker(file_handler, self.diff(t0, t_wait), marker_id = self.ONSET_ID, description = "Action period started")
        t_wait = self.at(t_epoch, self.t_rest + self.t_onset)
        self.wait_until(t_wait)

        # Release period
        q_RCOM_PRO.put('release')
        print('RELEASE')
        self.put_marker_to_queue(send_queues_dict=send_queues_dict, marker_id = self.REL_ID)
        self.log_marker(file_handler, self.diff(t0, t_wait), marker_id = self.REL_ID, description = "Release period started")
        t_wait = self.at(t_epoch, self.t_rest + self.t_onset + self.t_rel)
        self.wait_until(t_wait)

        if epoch_idx == self.num_epochs - 1:
            print("[OK] Experimental protocol completed.")
            return True

        print(f"Trial {epoch_idx + 1}/{self.num_epochs}")
        return False
    
    def execute_trim_period(self,
                            t0 : int,
                            file_handler,
                            send_queues_dict : dict):
        print('PROTOCOL - execute_trim_period func')
        current_time = time.perf_counter_ns()
        
        self.put_marker_to_queue(send_queues_dict, self.ST_TRIM_ID)
        self.log_marker(file_handler, self.diff(t0, current_time), marker_id = self.ST_TRIM_ID, description = "Trash data in the TRIM period")

        wait_for = self.at(current_time, self.t_trim)
        self.wait_until(wait_for)

        self.put_marker_to_queue(send_queues_dict, self.END_TRIM_ID)
        self.log_marker(file_handler, self.diff(t0, wait_for), marker_id = self.END_TRIM_ID, description = "Trash data in the TRIM period")

    def initialize_soundplay(self):
        '''
        initialize pygame and extract sound recordings to CUE events

        Returns
        -------
        CONTRACT_sound : pygame
        RELASE_sound : pygame
        REST_sound : pygame
        '''
        #-------------------#
        # Extract directory #
        #-------------------#
        path_dir = os.path.dirname(os.path.abspath(__file__))
        CON_dir = path_dir + "/beep_sounds" + "/CONTRACT_MOD.mp3"
        REL_dir = path_dir + "/beep_sounds" + "/RELEASE_MOD.mp3"
        RES_dir = path_dir + "/beep_sounds" + "/REST_MOD.mp3"

        #-----------------#
        # Initlize pygame #
        #-----------------#
        pygame.mixer.pre_init(
            frequency=44100,    # 
            size=-16,           # Bits used in the audio
            channels=2,         # Stero
            buffer=256          # lower buffer = lower latency - before 256
            )
        pygame.init()

        #---------------------#
        # Configure soundplay #
        #---------------------#
        CONTRACT_sound = pygame.mixer.Sound(CON_dir)
        RELEASE_sound = pygame.mixer.Sound(REL_dir)
        REST_sound = pygame.mixer.Sound(RES_dir)

        return CONTRACT_sound, RELEASE_sound, REST_sound


    def put_marker_to_queue(self,
                        send_queues_dict : dict,
                        marker_id : int):
        '''
        Put marker to all available queues in the send_queues_dict
        '''
        for key, queue in send_queues_dict.items():
            if queue is not None:
                queue.put( marker_id )

    def create_file_header(self, filepath):
        # if file doesn't exist, write header

        if os.path.exists(filepath):
            raise FileExistsError(f"File already exists: {filepath}")

        with open(filepath, 'w', newline='') as f:
            np.savetxt(f, np.array([["time", "marker_id", "description"]]),
                    delimiter=',', fmt='%s')

    def wait_until(self, t_deadline):
            '''
            Absolute wait using monotonic clock to avoid cumulative drift
            '''
            while True:
                now = time.perf_counter_ns()
                remaining = t_deadline - now
                if remaining <= 0:
                    return
                if remaining > 2_000_000:                               # sleep until we're ~2 ms away from the deadline (1 ms = 1_000_000 ns)
                    time.sleep((remaining - 2_000_000) / 1e9)           # Convert the remaining time to secounds
                else:
                    # tight spin for the last ~2 ms
                    while time.perf_counter_ns() < t_deadline:
                        pass
                    break

    def at(self, t_epoch, sec):
        return t_epoch + int(sec * 1e9)

    def diff(self, t_start, t_end):
        return (t_end - t_start) / 1e9  # return difference in seconds

    def log_marker(self, file_handler, time, marker_id, description=""):
        '''
        Create serperate log file for markers
        args:
            time: timestamp in nanoseconds given by at function
            marker_id: marker code
            description: optional description of the marker
        '''
        formatted_data = np.array([[time, marker_id, description]], dtype=str)
        np.savetxt(file_handler, formatted_data, delimiter=',', fmt='%s')

if __name__ == "__main__":
    
    
    t0 = time.perf_counter_ns()
    
    #CON_SOUND.play()
    print(((time.perf_counter_ns() - t0) / 1e9))
    print('CONTRACT')
    print(((time.perf_counter_ns() - t0) / 1e9))
    time.sleep(2)

    #REL_SOUND.play()
    print(((time.perf_counter_ns() - t0) / 1e9))
    print('RELASE')
    print(((time.perf_counter_ns() - t0) / 1e9))
    time.sleep(2)

    #RES_SOUND.play()
    print(((time.perf_counter_ns() - t0) / 1e9))
    print('REST')
    print(((time.perf_counter_ns() - t0) / 1e9))
    time.sleep(1)

