"""
Minimal experiment system with a visual cue GUI.

Two processes:
1) Main experiment logic (this file)
2) GUI process that only displays cues

Communication is done through a multiprocessing Queue.
"""
import os
import sys
import time
from multiprocessing import Queue, Process
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QStackedWidget
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import Qt, QTimer, QUrl
from typing import Any

##===========#
# GUI WINDOW #
##===========#
class CueWindow(QWidget):
    """
    Fullscreen window that displays the current cue:
    'REST', 'CONTRACT', or 'RELEASE'.
    """
    def __init__(self, event_queue : Queue, which_finger : str):
        super().__init__()
        self.queue = event_queue
        self.TEXT_IDX = 0
        self.VIDEO_IDX = 1
        self.widgets = ['switch_to_text', 'switch_to_video']        # Defined cmd to change between video and text

        # QT uses DirectShow by default, but windows can't provide a 
        # working DirectShow for Qt. Force Qt to use Media Foundation.
        os.environ["QT_MULTIMEDIA_PREFERRED_PLUGINS"] = "windowsmediafoundation"

        # Window setup
        self.setWindowTitle('Experiment Cues')          # Name of the window
        #self.setFixedSize(1000, 700)                    # Set window size
        self.center_stack = QStackedWidget(self)        # Allow for stackable widgets
        self.showFullScreen()

        #==============#
        # Video Widget #
        #==============#
        self.video_widget = QVideoWidget(self)

        self.player = QMediaPlayer(self)
        self.player.setVideoOutput(self.video_widget)
        
        video_path = os.path.abspath(f"src/utilities/hand_motion_{which_finger}.mp4")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path} - Ensure you are in the hybrid_bci working dictionary")
        
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
        
        # Define video positions (ms)
        self.positions = {
            'rest' : 0, 
            'contract' : 2000,
            'release' : 6000
        }
        
        #===========#
        # Top label #
        #===========#
        self.top_label = QLabel(f'Experiment for finger: {which_finger}', self)
        self.top_label.setAlignment(Qt.AlignCenter)
        self.top_label.setStyleSheet("""
            QLabel {
                font-size: 40px;
                color: lightgray;
                background-color: black;
            }
        """)

        #============================#
        # Bottom label (Show states) #
        #============================#
        self.bottom_label = QLabel('Initilize Window', self)
        self.bottom_label.setAlignment(Qt.AlignCenter)
        self.bottom_label.setStyleSheet("""
            QLabel {
                font-size: 60px;
                color: white;
                background-color: black;
            }
        """)
        self.bottom_label.hide()

        #=================================#
        # Overlay label (on top of video) #
        #=================================#
        self.overlay_label = QLabel('Hallo there...', self.video_widget)
        self.overlay_label.setAlignment(Qt.AlignCenter)
        self.overlay_label.setStyleSheet("""
            QLabel {
                font-size: 72px;
                color: white;
                background-color: black
            }
        """)
        #self.overlay_label.resize(self.video_widget.size())
        #self.overlay_label.show()
        
        #========#
        # Layout #
        #========#
        self.center_stack.addWidget(self.overlay_label) # Index 0
        self.center_stack.addWidget(self.video_widget)  # Index 1
        layout = QVBoxLayout()
        layout.addWidget(self.top_label, stretch = 1)
        layout.addWidget(self.center_stack, stretch = 4)
        layout.addWidget(self.bottom_label, stretch = 1)
        self.setLayout(layout)

        #==============================================================================#
        # Purpose: Timer checks for new events.                                        #
        # How to calculate the .start(x) value                                         #
        # 1/fs = value (s) convert to (ms)                                             #
        # Example, call check_event 60 times per second: 1/60Hz = 0.0166 s -> 16.6 ms. #
        #==============================================================================#
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_evnet)
        self.timer.start(16)  
    
    def switch_widget(self, cmd : str, payload : str):
        if cmd == self.widgets[self.VIDEO_IDX]:        # show video
            self.bottom_label.show()
            self.center_stack.setCurrentIndex(self.VIDEO_IDX)

        elif cmd == self.widgets[self.TEXT_IDX]:      # show text
            self.bottom_label.hide()
            self.overlay_label.setText(payload)
            self.center_stack.setCurrentIndex(self.TEXT_IDX)

    
    def stop_gui(self):
        self.timer.stop()
        self.player.stop()
        QApplication.quit()

    def check_evnet(self):
        """
        Poll the queue and update the displayed cue if a new message arrives.
        """
        while not self.queue.empty():                           # Exits when queue is empty
            msg = self.queue.get()                              # Fetch message in queue, which is a state -> [Rest, Contract, Release, other]

            if isinstance(msg, tuple):
                cmd, payload = msg
            else:
                cmd, payload = msg, None
            
            cmd = str(cmd).strip().lower()

            if cmd in self.positions.keys():                    # Check is message belong to desired states
                
                self.bottom_label.setText(cmd)                  # Change label text
                self.player.setPosition(self.positions[cmd])    # Set the video recording postion at desired time
                self.player.play()                              # Play the video
            
            elif cmd in self.widgets:
                self.switch_widget(cmd, payload)

            elif cmd == 'terminate':
                self.stop_gui()

#=========================#
# GUI PROCESS ENTRY POINT #
#=========================#
def run_gui(event_queue: Queue, which_finger : str, barrier_init : Any | None):
    '''
    Initialize the gui window, fetch video recording
    
    Parameter
    ---------
    event_queue : Queue
        Set a queue for messages to indicate states of motion
    
    which_finger : str
        Displays label text in gui and call corresponding video recording.\n
        The syntax should be of flex_<>_finger, where <> is desired finger.
    
    barrier_init : Barrier | None
        Ensure the gui initialization is synchronized with other processes.\n
        Provide 'None' to avoid barrier. Intended usage in cue_gui.show_finger_motion()
    '''
    selected_finger = which_finger.strip().lower().split('_')[1]
    if selected_finger not in ['thumb', 'index', 'middle', 'ring', 'little']:
        raise ValueError('which finger second word does not belong to a finger type')
    
    app = QApplication(sys.argv)
    win = CueWindow(event_queue, selected_finger)
    win.show()

    if barrier_init is not None:    # Avoid break if barrier_init is of None type
        barrier_init.wait()
    print('GUI - Starting process.')
    
    sys.exit(app.exec_())

def show_finger_motion():
    '''
    Prepare subjects on the finger motion by visualizing the speed and type of movement
    '''
    event_queue = Queue(100)
    WHICH_FINGER = 'flex_index_motion'

    gui_process = Process(target = run_gui, args = (event_queue, WHICH_FINGER, None))
    gui_process.start()

    cmd_switch_text = 'switch_to_text'
    cmd_switch_video = 'switch_to_video'
    states = [('rest', 2), ('contract', 4), ('release', 2)]
    EPOCHS = 2
    TIMEOUT = 3
    time.sleep(TIMEOUT + 3)               # Prepare the gui/video before sending messages

    for i in range(TIMEOUT):
        payload = f'Begins in {3-i}'
        cmd = cmd_switch_text
        event_queue.put((cmd, payload))
        time.sleep(1)

    event_queue.put(('switch_to_text', 'REST YOUR HAND'))
    time.sleep(TIMEOUT)
    event_queue.put(cmd_switch_video)

    for _ in range(EPOCHS):
        for state, t in states:
            event_queue.put(state)
            time.sleep(t)

    event_queue.put((cmd_switch_text, 'GOOD JOB!'))
    time.sleep(TIMEOUT)
    event_queue.put('terminate')

if __name__ == '__main__':
    show_finger_motion()
