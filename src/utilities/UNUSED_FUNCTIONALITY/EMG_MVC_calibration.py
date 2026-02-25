from .pytrigno import MVC
from .create_record_folders import create_recording_folder
from pathlib import Path
import json

SUBJECT_NAME = "Nicklas_basic_movement"                                   # Define the subject name                  
current_dir = Path(__file__).resolve().parent              # folder of current script
parent_dir = current_dir.parent
BASE_PATH = str(parent_dir) + r'\dynamixel_experiment\data'    # Define the base path for data storage

channel_names =[
'ch1 : Flexor digitorum superficialis',
'ch2 : Flexor pollicis longus'
]

# channel_names =[
# 'ch1 : Flexor digitorum superficialis',
# 'ch2 : Flexor pollicis longus', 
# 'ch3 : Extensor digitorum',
# 'ch4 : Extensor pollicis brevis/longus'
# ]

''' How to calibrate new data:
normalized_data = (raw_data - baseline_noise) / (MVC - baseline_noise) 
'''


def main():
    folders = create_recording_folder(SUBJECT_NAME=SUBJECT_NAME, BASE_PATH=BASE_PATH)
    calibration_path = folders['BASE'] / 'calibration_stats.json'

    if calibration_path.exists():
        raise FileExistsError(f'Calibration file already exists : {calibration_path}')

    mvc = MVC(channel_range = (0, 1), samples_per_read = 200, units = 'mV')

    baseline_noise, mvc = mvc.start_mvc_protocol(rest_window_sec = 6,
                                                 contract_window_sec = 3,
                                                 repetition = 5)
    
    calibration = {}
    for i, ch in enumerate(channel_names):
        calibration[ch] = {
            'baseline_noise': float(baseline_noise[i]),
            'MVC': float(mvc[i])
        }
        
    with open(calibration_path, 'w') as f:
        json.dump(calibration, f, indent = 4)
    
    print('[Success] - calibration performed and written to file')

if __name__ == '__main__':
    main()

