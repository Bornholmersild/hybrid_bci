# Manage path and arguments
from pathlib import Path

# Manage structure
import multiprocessing
import threading
from dataclasses import dataclass

# Manage time
from datetime import datetime
import time

# Own implementation
# from ..utilities.dynamixel_control import main as MC
# from .EMG_collector import EMG_con as EMG
# from .EEG_collector import EEG_con as EEG
from ..utilities.create_record_folders import create_recording_folder
from ..experiment.experimental_protocol import execute_protocol
       
current_dir = Path(__file__).resolve().parent     # folder of current script
parent_dir = current_dir.parent                   # one level up

#-----------#
# Constants #
#-----------#
METHOD = 'MC _ EMG'                                     # Select method in MODES by filled out blank: _ _ _. Where 'all' -> MC EEG EMG
BASE_PATH = str(parent_dir) + r'\experiment\data'       # Where to store DATA
SUBJECT_NAME = "data_fusion_simplifier"                 # Name of the subject : subject 0, subject 1
FINGER_NAME = 'flex_index_finger'                       # Name of the data file
NUM_EPOCHS = 30                                         # Number of epochs per experiment
REST_DURATION = 2.0                                     # Rest duration (sec) during 1 trial
ONSET_DURATION = 3.0                                    # ONSET duration (sec) during 1 trial
REL_DURATION = 2.0                                      # Release duration (sec) during 1 trial
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

# def MC_start(q_MC, q_ICOM_MC, q_RCOM_MC, barrier_init, barrier_execute):
#     mc = MC.Ada_con(co_mod = 0, re_only = False, devicename = MC_PORT)  # linux: '/dev/ttyUSB0', windows: 'COM3'
#     barrier_init.wait()
#     print('MC - [OK] Starting MC process.')
#     mc.start(q_MC, q_ICOM_MC, q_RCOM_MC, barrier_execute)
#     mc.close()

# def EEG_start(q_EEG, q_ICOM_EEG, q_RCOM_EEG, barrier_init, barrier_execute):
#     ac = EEG(serial_port = EEG_PORT)  # BEWARE OF SERIAL_PORT
#     barrier_init.wait()
#     ac.start(q_EEG, q_ICOM_EEG, q_RCOM_EEG, barrier_execute)

# def EMG_start(q_EMG, q_ICOM_EMG, q_RCOM_EMG, barrier_init, barrier_execute):
#     emg = EMG(select_sensors = EMG_SELECT_SENSORS, samples_per_read = EMG_SAMPLES_PER_READ, units = 'mV')
#     barrier_init.wait()
#     emg.start(q_EMG, q_ICOM_EMG, q_RCOM_EMG, barrier_execute)
def MC_start(q_MC, q_ICOM_MC, q_RCOM_MC, barrier_init, barrier_execute):
    print('MC - wait barrier init')
    barrier_init.wait()
    print('MC - Starting MC process.')
    
    while True:
            if not q_ICOM_MC.empty():      # Enter if there is a queue
                instruction = q_ICOM_MC.get()  # Get the instruction from the queue
                match instruction[0]:
                    case "record":
                        print('MC : queue received')
                        
                        print('MC - Waiting for barrier')
                        barrier_execute.wait()        
                        print('MC - OUT OF BARRIER')

                    case "stop":
                        print('MC - Stopping EEG recording')
                        break

def EEG_start(q_EEG, q_ICOM_EEG, q_RCOM_EEG, barrier_init, barrier_execute):
    print('EEG - wait barrier init')
    barrier_init.wait()
    print('EEG - Starting MC process.')
    
    while True:
            if not q_ICOM_EEG.empty():      # Enter if there is a queue
                instruction = q_ICOM_EEG.get()  # Get the instruction from the queue
                match instruction[0]:
                    case "record":
                        print('EEG : queue received')
                        time.sleep(3)
                        print('EEG - Waiting for barrier')
                        barrier_execute.wait()  
                        print('EEG - OUT OF BARRIER')      
               
                    case "stop":
                        print('EEG - Stopping EEG recording')
                        break

def EMG_start(q_EMG, q_ICOM_EMG, q_RCOM_EMG, barrier_init, barrier_execute):
    print('EMG - wait barrier init')
    barrier_init.wait()
    print('EMG - Starting MC process.')
    
    while True:
            if not q_ICOM_EMG.empty():      # Enter if there is a queue
                instruction = q_ICOM_EMG.get()  # Get the instruction from the queue
                match instruction[0]:
                    case "record":
                        print('EMG : queue received')
                        time.sleep(6)
                        print('EMG - Waiting for barrier')
                        barrier_execute.wait()        
                        print('EMG - OUT OF BARRIER')

                    case "stop":
                        print('EMG - Stopping EEG recording')
                        break

def PROTOCOL_start(q_PRO, q_i_PRO, q_r_PRO, barrier_init, barrier_execute):
        while True:
            if not q_i_PRO.empty():      # Enter if there is a queue
                instruction = q_i_PRO.get()  # Get the instruction from the queue
                match instruction[0]:
                    case "record":
                        print('PRO : queue received')
                        time.sleep(9)
                        print('PRO - Waiting for barrier')
                        barrier_execute.wait()        
                        print('PRO - OUT OF BARRIER')

                    case "stop":
                        print('PRO - Stopping EEG recording')
                        break

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

def listen_for_terminal_input(q_i_MC, q_i_EEG, q_i_EMG, q_i_PRO, barrier_init, select_method = None):
    """Listen for terminal input and send commands to the queue."""
    barrier_init.wait()
    while True:
        command = input("\nEnter command (e.g., 'record', 'stop'):\n ").strip().lower()
        
        if command == "record":

            folders = create_recording_folder(SUBJECT_NAME, BASE_PATH)

            current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

            instructions = []
            for key in ['MC', 'EEG', 'EMG', 'Markers']:
                filepath = str(folders[key]) + f"/{FINGER_NAME}_{current_time}.csv" 
                instruction_temp = (command, filepath)
                instructions.append(instruction_temp)
            
            send_command_queue(q_i_MC, q_i_EEG, q_i_EMG, q_i_PRO, instructions, select_method)

        elif command == "stop":
            instructions = [('stop', None), ('stop', None), ('stop', None), ('stop', None)]
            send_command_queue(q_i_MC, q_i_EEG, q_i_EMG, q_i_PRO, instructions, select_method)
            break

        else:
            print("Unknown command. Please enter 'record' or 'stop'.")

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
    barrier_init = multiprocessing.Barrier(num_sensors)         # Purpose: To hold processes until all is initilized 
    barrier_exec = multiprocessing.Barrier(num_sensors)         # Purpose: To hold processes to insure all is syncronized

    for key in active_sensors:
        # Queues for inter-process communications
        q_main = multiprocessing.JoinableQueue(100)                 # Purpose: Queue to main process
        q_i = multiprocessing.JoinableQueue(100)                    # Purpose: Queue to receive instructions from main process
        q_r = multiprocessing.JoinableQueue(100)                    # Purpose: Queue to send responses back to main process

        queues[key] = (q_main, q_i, q_r)                            # Load queues into dict with key-ID

        process_temp = multiprocessing.Process(
            target = SENSORS[key].start_func,
            args = (q_main, q_i, q_r, barrier_init, barrier_exec)
        )
        processes.append(process_temp)
    
    return queues, processes, barrier_init, barrier_exec

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

    queues, processes, barrier_init, barrier_exec = build_system(active)

    # What is [1] -> Get the q_i for each process.
    # If a process is not active, default set value (q_main, q_i, q_r) to None 
    q_MC = queues.get('MC', (None, None, None))[1]
    q_EEG = queues.get('EEG', (None, None, None))[1]
    q_EMG = queues.get('EMG', (None, None, None))[1]
    q_PRO = queues.get('PRO', (None, None, None))[1]

    terminal = threading.Thread(
        target = listen_for_terminal_input,
        args = (q_MC, q_EEG, q_EMG, q_PRO, barrier_init, METHOD),
        daemon = True
    )

    for p in processes:
        p.start()

    terminal.start()

if __name__ == '__main__':
    main()
