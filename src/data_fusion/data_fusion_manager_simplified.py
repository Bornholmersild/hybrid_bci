# Manage path and arguments
from pathlib import Path
import msvcrt

# Manage structure
from multiprocessing import JoinableQueue, Process, Barrier, Event
import threading
from dataclasses import dataclass
from typing import Dict, Optional

# Manage time
from datetime import datetime

# Own implementation
from ..utilities.dynamixel_control import main as MC
from .EMG_collector import EMG_con as EMG
from .EEG_collector import EEG_con as EEG
from ..utilities.create_record_folders import create_recording_folder
from ..experiment.experimental_protocol import PROTOCOL_con as PROTOCOL
       
current_dir = Path(__file__).resolve().parent     # folder of current script
parent_dir = current_dir.parent                   # one level up

#-----------#
# Constants #
#-----------#
METHOD = '_ EEG EMG'                                     # Select method in MODES by filled out blank: _ _ _. Where 'all' -> MC EEG EMG
BASE_PATH = str(parent_dir) + r'\experiment\data'       # Where to store DATA
SUBJECT_NAME = "data_fusion_simplifier"                 # Name of the subject : subject 0, subject 1
FINGER_NAME = 'flex_index_finger'                       # Name of the data file
NUM_EPOCHS = 1                                         # Number of epochs per experiment
REST_DURATION = 2                                       # Rest duration (sec) during 1 trial
ONSET_DURATION = 3                                      # ONSET duration (sec) during 1 trial
REL_DURATION = 2                                        # Release duration (sec) during 1 trial
TRIM_DURATION = 3                                       # Trim duration (sec) in the beginning and end of experiment
MC_PORT = 'COM8'                                        # Define MC port
EEG_PORT = 'COM9'                                       # Define EEG port
EMG_SELECT_SENSORS = (0, 2)                             # EMG data channels. For EMG only: sensor1 = 0, sensor2 = 1, sensor3 = 2
EMG_SAMPLES_PER_READ = 1000                             # Samples per read for the EMG sensors

''' BEFORE EXECUTION
* Note which COM port is in use
* EMG class : select_sensors and samples_per_read can be changed depending on usecase
* SUBJECT_NAME and BASE_PATH can be changed
* listen_for_terminal_input :
    execute_protocol : Change experiment protocol
    file_path : Can add description infront of 'current_time'. Like filepath_MC = folders["MC"] / f"Example_{current_time}.csv"
* Remember to check when processes start and it might need to be shifted
'''

def MC_start(q_MC, q_ICOM_MC, q_RCOM_MC, barrier_init, barrier_execute):
    mc_ins = MC.Ada_con(co_mod = 0, re_only = False, devicename = MC_PORT)  # linux: '/dev/ttyUSB0', windows: 'COM3'
    barrier_init.wait()
    print('MC - Starting process.')
    mc_ins.start(q_MC, q_ICOM_MC, q_RCOM_MC, barrier_execute)
    mc_ins.close()

def EEG_start(q_EEG, q_ICOM_EEG, q_RCOM_EEG, barrier_init, barrier_execute):
    eeg_ins = EEG(serial_port = EEG_PORT)  # BEWARE OF SERIAL_PORT
    barrier_init.wait()
    print('EEG - Starting process.')
    eeg_ins.start(q_EEG, q_ICOM_EEG, q_RCOM_EEG, barrier_execute)

def EMG_start(q_EMG, q_ICOM_EMG, q_RCOM_EMG, barrier_init, barrier_execute):
    emg_ins = EMG(select_sensors = EMG_SELECT_SENSORS, samples_per_read = EMG_SAMPLES_PER_READ, units = 'mV')
    barrier_init.wait()
    print('EMG - Starting process.')
    emg_ins.start(q_EMG, q_ICOM_EMG, q_RCOM_EMG, barrier_execute)

def PROTOCOL_start(q_PRO, q_i_PRO, q_r_PRO, barrier_init, barrier_execute, shutdown_event):
    protocol_ins = PROTOCOL(num_epochs = NUM_EPOCHS,
                            rest_duration = REST_DURATION,
                            onset_duration = ONSET_DURATION,
                            release_duration = REL_DURATION,
                            trim_duration = TRIM_DURATION)
    barrier_init.wait()
    print('PROTOCOL - Starting process.')
    protocol_ins.start(q_PRO, q_i_PRO, q_r_PRO, barrier_execute)
    
    shutdown_event.set()        # Set shutdown_event to terminate all processes

def send_command_queue(q_i_MC, q_i_EEG, q_i_EMG, q_i_PRO, instruction, method):
    active = MODES[method]
    mapping = {'MC' : q_i_MC,
               'EEG' : q_i_EEG,
               'EMG' : q_i_EMG,
               'PRO' : q_i_PRO
               }
    
    for i, key in enumerate(['MC', 'EEG', 'EMG', 'PRO']):
        if key in active:
            mapping[key].put(instruction[i])

def listen_for_terminal_input(q_i_MC : Optional[JoinableQueue],
                              q_i_EEG : Optional[JoinableQueue],
                              q_i_EMG : Optional[JoinableQueue],
                              q_i_PRO : Optional[JoinableQueue], 
                              barrier_init : any,
                              select_method : Dict,
                              shutdown_event : any):
    """Listen for terminal input and send commands to the queue."""
    barrier_init.wait()
    command = None
    
    print('Write "record" to start protocol and write "stop" to end execution')
    while not shutdown_event.is_set():
        
        if msvcrt.kbhit():                     # key pressed?
            command = input("\nEnter command: ").strip().lower()
        
        if command == "record":
            
            folders = create_recording_folder(SUBJECT_NAME, BASE_PATH)

            current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

            instructions = []
            for key in ['MC', 'EEG', 'EMG', 'Markers']:
                filepath = str(folders[key]) + f"/{FINGER_NAME}_{current_time}.csv" 
                instruction_temp = (command, filepath)
                instructions.append(instruction_temp)
            
            send_command_queue(q_i_MC, q_i_EEG, q_i_EMG, q_i_PRO, instructions, select_method)
            command = None      # Reset back to None

        elif command == "stop":
            instructions = [('stop', None)] * 4
            send_command_queue(q_i_MC, q_i_EEG, q_i_EMG, q_i_PRO, instructions, select_method)
            break

        elif command is not None:
            print("Unknown command. Please enter 'record' or 'stop'.")
            command = None
    
    if shutdown_event.is_set():
        instructions = [('stop', None)] * 4
        send_command_queue(q_i_MC, q_i_EEG, q_i_EMG, q_i_PRO, instructions, select_method)

def build_system(active_sensors : dict):
    """
    Build queues, processes, and barriers corresponding to the number of active sensors.

    Parameters
    ----------
    active_sensors : dict
        Dictionary describing the selected MODE and its active sensors.

    Returns
    -------
    queues : dict
        Mapping from sensor name to a tuple of communication queues:
        (q_main, q_ICOM, q_RCOM).
        Example:
            {
                "MC":  (JoinableQueue, JoinableQueue, JoinableQueue),
                "EEG": (JoinableQueue, JoinableQueue, JoinableQueue),
                ...
            }

    processes : list[multiprocessing.Process]
        List of all spawned sensor processes.

    barrier_init : multiprocessing.Barrier
        Barrier used to synchronize initialization of all sensor processes.

    barrier_exec : multiprocessing.Barrier
        Barrier used to synchronize execution across all sensor processes.
    """
    queues = {}
    processes = []

    num_sensors = len(active_sensors)
    barrier_init = Barrier(num_sensors + 1)     # Purpose: To hold processes until all is initilized + listen_for_terminal_input function
    barrier_exec = Barrier(num_sensors)         # Purpose: To hold processes to insure all is syncronized

    for key in active_sensors:
        # Queues for inter-process communications
        q_main = JoinableQueue(100)                 # Purpose: Queue to main process
        q_i = JoinableQueue(100)                    # Purpose: Queue to receive instructions from main process
        q_r = JoinableQueue(100)                    # Purpose: Queue to send responses back to main process

        queues[key] = (q_main, q_i, q_r)            # Load queues into dict with key-ID

        if key == 'PRO':
            shutdown_event = Event()                # Purpose: Whenever protocol terminates, set this true and it will terminate all processes
            args = (q_main, q_i, q_r, barrier_init, barrier_exec, shutdown_event)
        else:
            args = (q_main, q_i, q_r, barrier_init, barrier_exec)

        process_temp = Process(
            target = SENSORS[key].start_func,
            args = args
        )
        processes.append(process_temp)
    
    return queues, processes, barrier_init, shutdown_event

#---------------#
# Configuration #
#---------------#
@dataclass
class Sensor:
    name : str
    start_func : callable

SENSORS = {
    'MC' : Sensor('MC', MC_start),
    'EEG' : Sensor('EEG', EEG_start),
    'EMG' : Sensor('EMG', EMG_start),
    'PRO' : Sensor('PRO', PROTOCOL_start),
}

MODES = {
    "MC _ _":    ["MC", "PRO"],
    "MC EEG _":  ["MC", "EEG", "PRO"],
    "MC _ EMG":  ["MC", "EMG", "PRO"],
    "_ EEG _":   ["EEG", "PRO"],
    "_ EEG EMG": ["EEG", "EMG", "PRO"],
    "_ _ EMG":   ["EMG", "PRO"],
    "all":       ["MC", "EEG", "EMG", "PRO"],
}


def main():
    if METHOD not in MODES:
        raise ValueError('Invalid method')

    active = MODES[METHOD]             # Extract the mode from the desired argument

    queues, processes, barrier_init, shutdown_event  = build_system(active)

    # What is [1] -> Get the q_i for each process.
    # If a process is not active, default set value (q_main, q_i, q_r) to None 
    q_MC = queues.get('MC', (None, None, None))[1]
    q_EEG = queues.get('EEG', (None, None, None))[1]
    q_EMG = queues.get('EMG', (None, None, None))[1]
    q_PRO = queues.get('PRO', (None, None, None))[1]

    terminal = threading.Thread(
        target = listen_for_terminal_input,
        args = (q_MC, q_EEG, q_EMG, q_PRO, barrier_init, METHOD, shutdown_event),
        daemon = True
    )

    for p in processes:
        p.start()

    terminal.start()

if __name__ == '__main__':
    main()
