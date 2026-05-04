# Communication
import socket

# Syncronization
import time

# Own implementation
from src.experiment.experimental_protocol import PROTOCOL_con


class Exp1BendFingerProtocol(PROTOCOL_con):
    def __init__(self,
                 num_epochs : int,
                 rest_duration : int,
                 onset_duration : int,
                 release_duration : int,
                 trim_duration : int = 3):
        
        super().__init__(
            num_epochs=num_epochs,
            rest_duration=rest_duration,
            onset_duration=onset_duration,
            release_duration=release_duration,
            trim_duration=trim_duration
        )

        self.tcp_socket = None
        self.ticks = (1494 * 3) - 1                   # 1494 ticks correspnds to one rotation

        self.init_protocol(
            host = "10.126.128.129",  # <-- Replace with ESP IP from Serial Monitor
            port = 1234
        )

    def init_protocol(self, host, port):
        HOST = host  # <-- Replace with ESP IP from Serial Monitor
        PORT = port

        # Create TCP socket
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to ESP server
        self.tcp_socket.connect((HOST, PORT))

        # Confirm connection
        connected = False
        while not connected:

            self.tcp_socket.sendall(b'connect\n')

            scp_msg = self.tcp_socket.recv(1024).decode().strip()

            if scp_msg == 'connection complete':
                print('ESP connection established')
                connected = True
            else:
                print('Connecting...')
    
    def send_comment_esp(self, action : str):
        
        cmd = f"{action} {self.ticks}"
        
        try:
            msg = f"{cmd}\n".encode("utf-8")            # Raw bytes
            self.tcp_socket.sendall(msg)

        except Exception as e:
            print(f"ESP error: {e}")
    
    def execute_protocol(self, 
                         t0 : int,
                         epoch_idx : int,
                         file_handler,
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
        print(f"Trial {epoch_idx}/{self.num_epochs}")

        # Add some control here. 
        #------------#
        # Rest event #
        #------------#
        # self.RES_SOUND.play()
        print('REST')
        t_epoch = time.perf_counter_ns()
        self.put_marker_to_queue(send_queues_dict = send_queues_dict, marker_id = self.REST_ID)
        self.log_marker(file_handler, self.diff(t0, t_epoch), marker_id = self.REST_ID, description = "Rest period started")
        t_wait = self.at(t_epoch, self.t_rest)
        self.wait_until(t_wait)

        #----------------#
        # Contract event #
        #----------------#
        # self.CON_SOUND.play()
        self.send_comment_esp('contract')                # Move motor
        print('Contract')
        self.put_marker_to_queue(send_queues_dict=send_queues_dict, marker_id = self.ONSET_ID)
        self.log_marker(file_handler, self.diff(t0, t_wait), marker_id = self.ONSET_ID, description = "Action period started")
        t_wait = self.at(t_epoch, self.t_rest + self.t_onset)
        self.wait_until(t_wait)

        #---------------#
        # Release event #
        #---------------#
        # self.REL_SOUND.play()
        self.send_comment_esp('release')
        print('RELEASE')
        self.put_marker_to_queue(send_queues_dict=send_queues_dict, marker_id = self.REL_ID)
        self.log_marker(file_handler, self.diff(t0, t_wait), marker_id = self.REL_ID, description = "Release period started")
        t_wait = self.at(t_epoch, self.t_rest + self.t_onset + self.t_rel)
        self.wait_until(t_wait)

        if epoch_idx == self.num_epochs - 1:
            print("[OK] Experimental protocol completed.")
            return True

        return False

if __name__ == '__main__':
    print('It fine')


