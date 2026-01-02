from pathlib import Path

import multiprocessing
import threading
from datetime import datetime
import argparse
import time

from ..utilities.dynamixel_control import main as MC
from .EMG_collector import EMG_con as EMG
from .EEG_collector import EEG_con as EEG
from ..utilities.create_record_folders import create_recording_folder
from ..experiment.experimental_protocol import execute_protocol

SUBJECT_NAME = "Nicklas_wrist_EMG"                                   # Define the subject name                  
current_dir = Path(__file__).resolve().parent     # folder of current script
parent_dir = current_dir.parent                   # one level up
BASE_PATH = str(parent_dir) + r'\experiment\data'    # Define the base path for data storage

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
    mc = MC.Ada_con(co_mod = 0, re_only = False, devicename = 'COM8')  # linux: '/dev/ttyUSB0', windows: 'COM3'
    barrier_init.wait()
    print('MC - [OK] Starting MC process.')
    mc.start(q_MC, q_ICOM_MC, q_RCOM_MC, barrier_execute)
    mc.close()

def EEG_start(q_EEG, q_ICOM_EEG, q_RCOM_EEG, barrier_init, barrier_execute):
    ac = EEG(serial_port='COM9')  # BEWARE OF SERIAL_PORT
    barrier_init.wait()
    ac.start(q_EEG, q_ICOM_EEG, q_RCOM_EEG, barrier_execute)

def EMG_start(q_EMG, q_ICOM_EMG, q_RCOM_EMG, barrier_init, barrier_execute):
    emg = EMG(select_sensors = (0, 5), samples_per_read=1000, units = 'mV')
    barrier_init.wait()
    emg.start(q_EMG, q_ICOM_EMG, q_RCOM_EMG, barrier_execute)

def send_command_queue(q_ICOM_MC, q_ICOM_EEG, q_ICOM_EMG, instruction, method):
    if method == 'MC EEG _':
        q_ICOM_MC.put(instruction[0])  
        q_ICOM_EEG.put(instruction[1])
    
    elif method == '_ _ EMG':
        q_ICOM_EMG.put(instruction[2])

    elif method == '_ EEG _':
        q_ICOM_EEG.put(instruction[1])
    
    elif method == 'MC _ _':
        q_ICOM_MC.put(instruction[0])

    elif method == 'MC _ EMG':
        q_ICOM_MC.put(instruction[0])  
        q_ICOM_EMG.put(instruction[2])
    
    elif method == '_ EEG EMG':
        q_ICOM_EEG.put(instruction[1])
        q_ICOM_EMG.put(instruction[2])

    elif method == 'all':
        q_ICOM_MC.put(instruction[0])  
        q_ICOM_EEG.put(instruction[1])
        q_ICOM_EMG.put(instruction[2])

def listen_for_terminal_input(q_ICOM_MC, q_ICOM_EEG, q_ICOM_EMG, barrier_init, barrier_execute, select_method = None):
    """Listen for terminal input and send commands to the queue."""
    barrier_init.wait()
    while True:
        command = input("\nEnter command (e.g., 'record', 'stop'):\n ").strip().lower()
        
        if command == "record":

            folders = create_recording_folder(SUBJECT_NAME, BASE_PATH)

            current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            
            # The last two epochs of alongside_body was with closed eyes
            # The last five epochs of thumb was with closed eyes

            # Define file paths
            finger_name = 'test'
            filepath_MC = folders["MC"] / f"{finger_name}_{current_time}.csv"
            filepath_EEG = folders["EEG"] / f"{finger_name}_{current_time}.csv"
            filepath_EMG = folders["EMG"] / f"{finger_name}_{current_time}.csv"
            filepath_markers = folders["Markers"] / f"{finger_name}_{current_time}.csv"

            # Create instruction tuples
            instruction_MC = (command, filepath_MC)
            instruction_EEG = (command, filepath_EEG)
            instruction_EMG = (command, filepath_EMG)
            instructions = [instruction_MC, instruction_EEG, instruction_EMG]
            
            send_command_queue(q_ICOM_MC, q_ICOM_EEG, q_ICOM_EMG, instructions, select_method)

            execute_protocol(num_epochs=30, rest_duration=2.0, action_duration=3.0, release_duration=2.0, trim_epoch=0.0, filepath=filepath_markers, barrier=barrier_execute)

            instructions = [('stop', None), ('stop', None), ('stop', None)]
            send_command_queue(q_ICOM_MC, q_ICOM_EEG, q_ICOM_EMG, instructions, select_method)
            break

        elif command == "stop":
            instructions = [('stop', None), ('stop', None), ('stop', None)]
            send_command_queue(q_ICOM_MC, q_ICOM_EEG, q_ICOM_EMG, instructions, select_method)
            break

        else:
            print("Unknown command. Please enter 'record' or 'stop'.")

def main():

    # Other arguments
    # EMG samples
    # Subject name
    # Num trails
    # EMG select channels
    
    parser = argparse.ArgumentParser(description="Data Fusion Manager for biosignals")
    parser.add_argument("--method", type=str, required=True,
                        help="Select method to use: 'MC EEG _', 'MC _ EMG', '_ EEG EMG', or 'all'")

    args = parser.parse_args()

    # Use args values
    print(f"Recording subject: {args.method}")

    select_sensors = args.method              # Can be EEG, EMG, and BOTH

    if select_sensors == 'MC EEG _':

        #Queues for inter-process communications
        q_MC = multiprocessing.JoinableQueue(maxsize=100)
        q_ICOM_MC = multiprocessing.JoinableQueue(maxsize=100)
        q_RCOM_MC = multiprocessing.JoinableQueue(maxsize=100)

        q_EEG = multiprocessing.JoinableQueue(maxsize=100)
        q_ICOM_EEG = multiprocessing.JoinableQueue(maxsize=100)
        q_RCOM_EEG = multiprocessing.JoinableQueue(maxsize=100)

        barrier_init = multiprocessing.Barrier(3)
        barrier_execute = multiprocessing.Barrier(3)
        
        p_mc = multiprocessing.Process(target=MC_start, args=(q_MC, q_ICOM_MC, q_RCOM_MC, barrier_init, barrier_execute))
        p_eeg = multiprocessing.Process(target=EEG_start, args=(q_EEG, q_ICOM_EEG, q_RCOM_EEG, barrier_init, barrier_execute))

        terminal_thread = threading.Thread(target=listen_for_terminal_input, args=(q_ICOM_MC, q_ICOM_EEG, None, barrier_init, barrier_execute, select_sensors), daemon=True)
        
        p_mc.start()
        p_eeg.start()

    elif select_sensors == '_ _ EMG':
        #Queues for inter-process communications
        q_EMG = multiprocessing.JoinableQueue(maxsize=100)
        q_ICOM_EMG = multiprocessing.JoinableQueue(maxsize=100)
        q_RCOM_EMG = multiprocessing.JoinableQueue(maxsize=100)

        barrier_init = multiprocessing.Barrier(2)
        barrier_execute = multiprocessing.Barrier(2)

        p_emg = multiprocessing.Process(target=EMG_start, args=(q_EMG, q_ICOM_EMG, q_RCOM_EMG, barrier_init, barrier_execute))

        terminal_thread = threading.Thread(target=listen_for_terminal_input, args=(None, None, q_ICOM_EMG, barrier_init, barrier_execute, select_sensors), daemon=True)

        p_emg.start()

    elif select_sensors == '_ EEG _':
        #Queues for inter-process communications
        q_EEG = multiprocessing.JoinableQueue(maxsize=100)
        q_ICOM_EEG = multiprocessing.JoinableQueue(maxsize=100)
        q_RCOM_EEG = multiprocessing.JoinableQueue(maxsize=100)

        barrier_init = multiprocessing.Barrier(2)
        barrier_execute = multiprocessing.Barrier(2)

        p_eeg = multiprocessing.Process(target=EEG_start, args=(q_EEG, q_ICOM_EEG, q_RCOM_EEG, barrier_init, barrier_execute))

        terminal_thread = threading.Thread(target=listen_for_terminal_input, args=(None, q_ICOM_EEG, None, barrier_init, barrier_execute, select_sensors), daemon=True)

        p_eeg.start()

    elif select_sensors == 'MC _ _':
        #Queues for inter-process communications
        q_MC = multiprocessing.JoinableQueue(maxsize=100)
        q_ICOM_MC = multiprocessing.JoinableQueue(maxsize=100)
        q_RCOM_MC = multiprocessing.JoinableQueue(maxsize=100)

        barrier_init = multiprocessing.Barrier(2)
        barrier_execute = multiprocessing.Barrier(2)

        p_mc = multiprocessing.Process(target=MC_start, args=(q_MC, q_ICOM_MC, q_RCOM_MC, barrier_init, barrier_execute))

        terminal_thread = threading.Thread(target=listen_for_terminal_input, args=(q_ICOM_MC, None, None, barrier_init, barrier_execute, select_sensors), daemon=True)

        p_mc.start()

    elif select_sensors == 'MC _ EMG':
        #Queues for inter-process communications
        q_MC = multiprocessing.JoinableQueue(maxsize=100)
        q_ICOM_MC = multiprocessing.JoinableQueue(maxsize=100)
        q_RCOM_MC = multiprocessing.JoinableQueue(maxsize=100)

        q_EMG = multiprocessing.JoinableQueue(maxsize=100)
        q_ICOM_EMG = multiprocessing.JoinableQueue(maxsize=100)
        q_RCOM_EMG = multiprocessing.JoinableQueue(maxsize=100)

        barrier_init = multiprocessing.Barrier(3)
        barrier_execute = multiprocessing.Barrier(3)

        p_mc = multiprocessing.Process(target=MC_start, args=(q_MC, q_ICOM_MC, q_RCOM_MC, barrier_init, barrier_execute))
        p_emg = multiprocessing.Process(target=EMG_start, args=(q_EMG, q_ICOM_EMG, q_RCOM_EMG, barrier_init, barrier_execute))

        terminal_thread = threading.Thread(target=listen_for_terminal_input, args=(q_ICOM_MC, None, q_ICOM_EMG, barrier_init, barrier_execute, select_sensors), daemon=True)

        p_mc.start()
        time.sleep(3)
        p_emg.start()
    
    elif select_sensors == '_ EEG EMG':
        #Queues for inter-process communications
        q_EEG = multiprocessing.JoinableQueue(maxsize=100)
        q_ICOM_EEG = multiprocessing.JoinableQueue(maxsize=100)
        q_RCOM_EEG = multiprocessing.JoinableQueue(maxsize=100)

        q_EMG = multiprocessing.JoinableQueue(maxsize=100)
        q_ICOM_EMG = multiprocessing.JoinableQueue(maxsize=100)
        q_RCOM_EMG = multiprocessing.JoinableQueue(maxsize=100)

        barrier_init = multiprocessing.Barrier(3)
        barrier_execute = multiprocessing.Barrier(3)

        p_eeg = multiprocessing.Process(target=EEG_start, args=(q_EEG, q_ICOM_EEG, q_RCOM_EEG, barrier_init, barrier_execute))
        p_emg = multiprocessing.Process(target=EMG_start, args=(q_EMG, q_ICOM_EMG, q_RCOM_EMG, barrier_init, barrier_execute))

        terminal_thread = threading.Thread(target=listen_for_terminal_input, args=(None, q_ICOM_EEG, q_ICOM_EMG, barrier_init, barrier_execute, select_sensors), daemon=True)

        p_eeg.start()
        p_emg.start()
    
    elif select_sensors == 'all':
        #Queues for inter-process communications
        q_MC = multiprocessing.JoinableQueue(maxsize=100)
        q_ICOM_MC = multiprocessing.JoinableQueue(maxsize=100)
        q_RCOM_MC = multiprocessing.JoinableQueue(maxsize=100)

        q_EEG = multiprocessing.JoinableQueue(maxsize=100)
        q_ICOM_EEG = multiprocessing.JoinableQueue(maxsize=100)
        q_RCOM_EEG = multiprocessing.JoinableQueue(maxsize=100)

        q_EMG = multiprocessing.JoinableQueue(maxsize=100)
        q_ICOM_EMG = multiprocessing.JoinableQueue(maxsize=100)
        q_RCOM_EMG = multiprocessing.JoinableQueue(maxsize=100)

        barrier_init = multiprocessing.Barrier(4)
        barrier_execute = multiprocessing.Barrier(4)

        p_mc = multiprocessing.Process(target=MC_start, args=(q_MC, q_ICOM_MC, q_RCOM_MC, barrier_init, barrier_execute))
        p_eeg = multiprocessing.Process(target=EEG_start, args=(q_EEG, q_ICOM_EEG, q_RCOM_EEG, barrier_init, barrier_execute))
        p_emg = multiprocessing.Process(target=EMG_start, args=(q_EMG, q_ICOM_EMG, q_RCOM_EMG, barrier_init, barrier_execute))

        terminal_thread = threading.Thread(target=listen_for_terminal_input, args=(q_ICOM_MC, q_ICOM_EEG, q_ICOM_EMG, barrier_init, barrier_execute, select_sensors), daemon=True)

        p_mc.start()
        p_eeg.start()
        p_emg.start()
    
    else:
        raise TypeError(f'{select_sensors} is not valid argument. Use EEG, EMG or BOTH')

    terminal_thread.start()

if __name__ == '__main__':
    main()
