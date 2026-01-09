from multiprocessing import Process, JoinableQueue, Event, Barrier
import time
from ..utilities.cue_gui import run_gui

NUM_EPOCHS = 2                                         # Number of epochs per experiment
REST_DURATION = 2                                       # Rest duration (sec) during 1 trial
ONSET_DURATION = 4                                      # ONSET duration (sec) during 1 trial
REL_DURATION = 2                                        # Release duration (sec) during 1 trial
TRIM_DURATION = 3                                       # Trim duration (sec) in the beginning and end of experiment
STATES = ['REST', 'CONTRACT', 'RELEASE']
WHICH_FINGER = 'flex_index_finger'

def main():

    #======================================#
    # Initilize Queue, process for the gui #
    #======================================#
    #event_queue = Queue()
    barrier_init = Barrier(2)
    barrier_exec = Barrier(1)
    shutdown_event = Event()                # Purpose: Whenever protocol terminates, set this true and it will terminate all processes

    q_main = JoinableQueue(100)                 # Purpose: Queue to main process
    q_i = JoinableQueue(100)                    # Purpose: Queue to receive instructions from main process
    q_r = JoinableQueue(100)                    # Purpose: Queue to send responses back to main process

    protocol_process = Process(target = PROTOCOL_start, args = (q_main, q_i, q_r, shutdown_event, barrier_init, barrier_exec))
    gui_process = Process(target = run_gui, args=(q_r, WHICH_FINGER, barrier_init))

    gui_process.start()
    protocol_process.start()

    command = input('Enter command: ').strip().lower()
    q_i.put([command, None])

    while True:
        if shutdown_event.is_set():
            break

def PROTOCOL_start(q_PRO, q_i_PRO, q_r_PRO, shutdown_event, barrier_init, barrier_exec):
    protocol_ins = PROTOCOL_dummy_con(num_epochs = NUM_EPOCHS,
                            rest_duration = REST_DURATION,
                            onset_duration = ONSET_DURATION,
                            release_duration = REL_DURATION,
                            trim_duration = TRIM_DURATION)
    barrier_init.wait()
    print('PROTOCOL - Starting process.')
    protocol_ins.start(q_PRO, q_i_PRO, q_r_PRO, barrier_exec)
    
    shutdown_event.set()        # Set shutdown_event to terminate all processes
    print('PROTOCOL - Ending process')

class PROTOCOL_dummy_con():
    def __init__(self, 
                 num_epochs: int, 
                 rest_duration: int = 2, 
                 onset_duration: int = 4, 
                 release_duration: int = 2,
                 trim_duration : int = 3):

        self.REST_ID = 10
        self.ONSET_ID = 20
        self.REL_ID = 30
        self.END_ID = 40
        self.TRIM_ID = 111

        self.num_epochs = num_epochs
        self.t_rest = rest_duration
        self.t_onset = onset_duration
        self.t_rel = release_duration
        self.t_trim = trim_duration

    def start(self, 
               q_PRO : JoinableQueue,
               q_ICOM_PRO : JoinableQueue, 
               q_RCOM_PRO : JoinableQueue,
               barrier_exec : any):
        '''
        Calling this method listens for queue instructions and acts accordingly.
        '''
        execute_flag = False
        file_handle = None
        finish = False
        epoch_idx = 0
        t0 = 0

        while True:

            if not q_ICOM_PRO.empty():
                instruction = q_ICOM_PRO.get()

                match instruction[0]:
                    case 'record':
                        
                        print("PROTOCOL - [WAIT] WAITING for other processes. Dummy method.\n Begin in...")
                        for i in range(3):
                            payload = f'Begins in {3-i}'
                            cmd = 'switch_to_text'
                            q_RCOM_PRO.put((cmd, payload))
                            time.sleep(1)

                        barrier_exec.wait()
                        t0 = time.perf_counter_ns()
                        print(f'PROTOCOL - begin at time: {t0 / 1e9}')
                        
                        q_RCOM_PRO.put(('switch_to_text', 'REST YOUR HAND'))
                        self.execute_trim_period(t0 = t0, file_handler = None)
                        execute_flag = True
                        q_RCOM_PRO.put('switch_to_video')

                    case 'stop':
                        if not execute_flag:    # Break out of system, if 'record' never have been called
                            break
                        execute_flag = False    # Avoid executing the protocol 
                        finish = True           # Finish the experiment with trim period
            
            if execute_flag:
                finish = self.execute_protocol(t0 = t0, epoch_idx = epoch_idx, file_handler = file_handle, q_RCOM_PRO = q_RCOM_PRO)
                epoch_idx += 1

            if finish:
                q_RCOM_PRO.put(('switch_to_text', 'GREAT JOB!'))
                self.execute_trim_period(t0 = t0, file_handler = None)
                execute_flag, finish = False, False
                q_RCOM_PRO.put('terminate')
                break
    
    def execute_trim_period(self,
                            t0 : int,
                            file_handler):
        print('PROTOCOL - execute_trim_period func')
        current_time = time.perf_counter_ns()
        wait_for = self.at(current_time, self.t_trim)
        self.wait_until(wait_for)

    def execute_protocol(self, 
                         t0 : int,
                         epoch_idx : int,
                         file_handler,
                         q_RCOM_PRO : JoinableQueue):
        """Execute the experimental protocol.
        
        Args:
            num_epochs (int): Number of epochs to run
            rest_duration (float, optional): Duration of rest period in seconds. Defaults to 5.0.
            action_duration (float, optional): Duration of action period in seconds. Defaults to 5.0.
            release_duration (float, optional): Duration of release period in seconds. Defaults to 5.0.
            filepath (str, optional): Path to save markers. Defaults to None.
            barrier (multiprocessing.Barrier, optional): Synchronization barrier. Defaults to None.
        """
        print(STATES[0])
        t_epoch = time.perf_counter_ns()

        # Rest period
        q_RCOM_PRO.put(STATES[0])
        t_wait = self.at(t_epoch, self.t_rest)
        self.wait_until(t_wait)

        # Execute the action
        q_RCOM_PRO.put(STATES[1])
        print(STATES[1])
        t_wait = self.at(t_epoch, self.t_rest + self.t_onset)
        self.wait_until(t_wait)

        # Release period
        print(STATES[2])
        q_RCOM_PRO.put(STATES[2])
        t_wait = self.at(t_epoch, self.t_rest + self.t_onset + self.t_rel)
        self.wait_until(t_wait)

        if epoch_idx == self.num_epochs - 1:
            print("[OK] Experimental protocol completed.")
            return True

        print(f"Trial {epoch_idx + 1}/{self.num_epochs}")
        return False

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
    
if __name__ == '__main__':
    main()
    