import time
from pathlib import Path
from playsound3 import playsound
import numpy as np
from multiprocessing.synchronize import Barrier  # Import the proper Barrier type
import os

base_path = Path(__file__).resolve().parent  # folder where this script lives
beep_folder = str(base_path) + r"\beep_sounds"

CONTRACT_BEEP = beep_folder + r"\START.mp3"
RELEASE_BEEP = beep_folder + r"\RELEASE.mp3"

REST, CON, REL, TRIM_EPOCH, END = 10, 20, 30, 111, 000

def wait_until(t_deadline):
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

def at(t0, sec):
    return t0 + int(sec * 1e9)

def diff(t_start, t_end):
    return (t_end - t_start) / 1e9  # return difference in seconds

def log_marker(file_handler, time, marker_id, description=""):
    '''
    Create serperate log file for markers
    args:
        time: timestamp in nanoseconds given by at function
        marker_id: marker code
        description: optional description of the marker
    '''
    formatted_data = np.array([[time, marker_id, description]], dtype=str)
    np.savetxt(file_handler, formatted_data, delimiter=',', fmt='%s')

def create_file_header(filepath):
    # if file doesn't exist, write header

    if os.path.exists(filepath):
        raise FileExistsError(f"File already exists: {filepath}")


    with open(filepath, 'w', newline='') as f:
        np.savetxt(f, np.array([["time", "marker_id", "description"]]),
                delimiter=',', fmt='%s')

def execute_protocol(num_epochs: int, rest_duration: float = 5.0, action_duration: float = 5.0, 
                    release_duration: float = 5.0, filepath: str | None = None, 
                    barrier: Barrier | None = None):
    """Execute the experimental protocol.
    
    Args:
        num_epochs (int): Number of epochs to run
        rest_duration (float, optional): Duration of rest period in seconds. Defaults to 5.0.
        action_duration (float, optional): Duration of action period in seconds. Defaults to 5.0.
        release_duration (float, optional): Duration of release period in seconds. Defaults to 5.0.
        filepath (str, optional): Path to save markers. Defaults to None.
        barrier (multiprocessing.Barrier, optional): Synchronization barrier. Defaults to None.
    """
    create_file_header(filepath=filepath)
    file_handler = open(filepath, 'a', buffering = 1, newline='')

    print("PROTOCOL - [OK] Waiting for barrier.")
    if barrier is not None:
        barrier.wait()

    t0 = time.perf_counter_ns()
    print(f'PROTOCOL - begin at time: {t0 / 1e9}')
    print('REST')
    for epoch in range(num_epochs):
        t_epoch = time.perf_counter_ns()

        # Rest period
        #playsound(REST_BEEP, block=False)
        #print('REST')
        log_marker(file_handler, diff(t0, t_epoch), marker_id=REST, description="Rest period started")
        t1 = at(t_epoch, rest_duration)
        wait_until(t1)

        # Execute the action
        playsound(CONTRACT_BEEP, block=False)
        print('Contract')
        log_marker(file_handler, diff(t0, t1), marker_id=CON, description="Action period started")
        t2 = at(t_epoch, rest_duration + action_duration)
        wait_until(t2)

        # Rest period
        playsound(RELEASE_BEEP, block=False)
        print('RELEASE')
        log_marker(file_handler, diff(t0, t2), marker_id=REL, description="Rest period started")
        t3 = at(t_epoch, rest_duration + action_duration + release_duration)
        wait_until(t3)

        # Additional data for epoch trim
        #log_marker(file_handler, diff(t0, t3), marker_id=TRIM_EPOCH, description="Trash data can be used for trim")
        #t4 = at(t_epoch, rest_duration + action_duration + release_duration + trim_epoch)
        #wait_until(t4)

        if epoch == num_epochs - 1:
            print("[OK] Experimental protocol completed.")
            log_marker(file_handler, diff(t0, t3), marker_id=END, description="Experiment ended")
            file_handler.close()

        print(f"Trial {epoch + 1}/{num_epochs}")


if __name__ == "__main__":
    execute_protocol(num_epochs=3, rest_duration=1.0, action_duration=2.0, release_duration=3.0, filepath="test_markers.csv")
    print("Beep sound played.")