# Manage datasets
import numpy as np

# Manage file paths
from pathlib import Path
import os

# Syncronization 
from time import time, perf_counter, sleep

# Mujoco
from myosuite.utils import gym
import msvcrt

# Own implementations
from src.experiment.real_time_operation import EMGRealTime, Buffer, Model, StateLogic
from src.models.classification_pipeline import EMGStreamProcessor

#==================#
# Global variables #
#==================#
EMG_FREQ = 2000
EEG_FREQ = 125
RMS_FREQ = 40                   # 40 for 500 samples, 125 for 32 samples (window)

EEG_USEABLE_CHANNELS = [2, 3, 6, 7, 8, 9, 10, 11]

EMG_LOWCUT = 20
EMG_HIGHCUT = 450
EEG_LOWCUT = 0.5
EEG_HIGHCUT = 30

EEG_NUM_CH = len(EEG_USEABLE_CHANNELS)
EMG_NUM_CH = 3

RMS_SAMPLING_WINDOW = 200           # 500 samples - 250 ms                      32 samples - 16 ms                                       
RMS_WINDOW_STEPSIZE = 25            # 50 samples - 25 ms (90 % overlap)         16 samples - 8 ms (50 % overlap)

HAMPEL_WINDOWSIZE = 100
HAMPEL_SIGMA = 3                    # Usually 2

SLIDING_WINDOW_SAMPLES = 1000
SLIDING_WINDOW_STEPSIZE = 200

EMG_SELECT_SENSORS = (0, 2)
EMG_SAMPLES_PER_READ = 200

state = "REST"

EMG_CONFIG_DICT = {
    'rms_windowsize' : RMS_SAMPLING_WINDOW,
    'rms_stepsize' : RMS_WINDOW_STEPSIZE,
    'hampel_windowsize' : HAMPEL_WINDOWSIZE,
    'hampel_sigma' : HAMPEL_SIGMA,
    'hampel_plot_option' : [False, None],
    'include_EMG' : False
}

REJECT_CONFIG_DICT = {
    'EEG_epoch_rejection_tolerance' : 6,
    'EMG_epoch_rejection_tolerance' : 6,
    'EEG_ch_acceptance' : 0,
    'EMG_ch_acceptance' : 0
}


def prepare_joints(mujoco_model, env_data, env, actions, THUMB_JOINT):
    #======================#
    # Constrain all joints #
    #======================#

    original_joint_ranges = mujoco_model.jnt_range.copy()        # save original joint limits

    for i in range(mujoco_model.njnt):
            q = mujoco_model.jnt_qposadr[i]
            current = env_data.qpos[q]

            mujoco_model.jnt_range[i] = [current, current]       # Lock joints in place 


    #================================#
    # Move thumb to desired position #
    #================================#
    _unfreeze_joints(model = mujoco_model, joints_list = THUMB_JOINT, OJR = original_joint_ranges)     
    actions[22], actions[23] = 0.3, 0.1         # Move thumb

    t0 = time()
    try: 
        while env_data.qpos[4] > -0.6:
            env.mj_render()                       # Render the current simulation frame
                
            env.step(actions)    
            
            if time() - t0 > 5:
                raise TimeoutError('thumb location not found')
    finally:
        actions[22], actions[23] = 0, 0

        for i in range(mujoco_model.njnt):
            q = mujoco_model.jnt_qposadr[i]
            current = env_data.qpos[q]

            mujoco_model.jnt_range[i] = [current, current]
        
        freezed_joint_ranges = mujoco_model.jnt_range.copy()        # save freezed joints ranges
    
    return freezed_joint_ranges, original_joint_ranges

def start_simulation(model_folder_name):
    #========#
    # Mujoco #
    #========#
    env = gym.make("myoHandPoseFixed-v0")
    env.reset()

    mujoco_model = env.sim.model
    env_data = env.sim.data

    NoC = mujoco_model.nu                         # Number of controls
    actions = np.zeros(NoC)                # Control vector with actuators

    INDEX_JOINT = [7,8,9,10]
    THUMB_JOINT = [3,4,5,6]
    CMC_FLAG = False
    MCP_FLAG = False

    freezed_joint_ranges, original_joint_ranges = prepare_joints(mujoco_model = mujoco_model,
                                                                env_data = env_data,
                                                                env = env,
                                                                actions = actions,
                                                                THUMB_JOINT = THUMB_JOINT)
        
    mcp_lower, mcp_upper, pm_upper = env_data.qpos[7], 1.59, 1.59
    cmc_lower, cmc_upper = env_data.qpos[4], -0.56              # Lower is fully flexed, upper is fully bend
    
    #=====================#
    # Prediction pipeline #
    #=====================#
    model_path_folder = Path(__file__).resolve().parents[1] / f"models/loggings/real_time/{model_folder_name}"

    MODEL = Model(path_to_model = model_path_folder)

    STREAM = EMGRealTime(config_dict = EMG_CONFIG_DICT,
                      select_sensors = EMG_SELECT_SENSORS,
                      samples_per_read = EMG_SAMPLES_PER_READ)
    
    EMG_BUFFER = Buffer(max_size = 10000,
                        num_ch = EMG_NUM_CH,
                        window_size = SLIDING_WINDOW_SAMPLES,
                        step_size = SLIDING_WINDOW_STEPSIZE)
    
    PREPROCESS = EMGStreamProcessor(fs = EMG_FREQ, lowcut = EMG_LOWCUT, highcut = EMG_HIGHCUT,
                                    reject_config_dict = EMG_CONFIG_DICT, 
                                    rms_window = RMS_SAMPLING_WINDOW, rms_step = RMS_WINDOW_STEPSIZE,
                                    hampel_window = HAMPEL_WINDOWSIZE, hampel_sigma = HAMPEL_SIGMA,     # sigma usually 2
                                    base_dir = 'Unused')
    
    STATE = StateLogic()
    # mu = np.load(model_path_folder / "mu.npy")
    # sigma = np.load(model_path_folder / "sigma.npy")

    
    STREAM.start_stream()                      # Initilize streaming
    
    buffer_fill_size = 1

    try:
        while True:    
            X_emg = STREAM.extract_data()               # Read data                        
            EMG_BUFFER.add_data(data = X_emg)           # Load into circular buffer
            
            if buffer_fill_size < 5:
                buffer_fill_size += 1
                continue
            
            X_win = EMG_BUFFER.get_window()             # Extract window of data by sliding window
            X_pre = PREPROCESS.update(chunk = X_win)    # Preprocess window of data

            if X_pre is None:
                print('Pre is none')
                continue
            
            # X_norm = (X_pre - mu) / (sigma + 1e-8)    # Normalize
            X_pred, confidence = MODEL.predict(input_data = X_pre)      # Insert into model
            state = STATE.update(pred = X_pred, confidence = confidence)    # Output of the model
            print(
            f"STATE: {state:<15} | "
            f"PRED: {X_pred:<20} | "
            f"CONF: {confidence:>6.2f}",
            end="\r"
            )
            
            actions = _actuate_motors(model = mujoco_model, command = state, actions = actions, original_joint_ranges = original_joint_ranges)

            env.mj_render()                       # Render the current simulation frame
            
            env.step(actions)                        # performs a physics step

            mcp = env_data.qpos[7]
            cmc = env_data.qpos[4]
            
            if cmc >= cmc_upper and not CMC_FLAG:           # Freeze thumb joints when fully bend
                print('[Debug] enter cmc1')
                actions[:] = 0
                
                _freeze_joints(model = mujoco_model, data=env_data, joints_list = THUMB_JOINT, current_joints = mujoco_model.jnt_range)
                CMC_FLAG = True

            elif cmc <= cmc_lower and CMC_FLAG:             # Disable activation when fully flexed and go to initial position
                print('[Debug] enter cmc2')
                actions[:] = 0
                CMC_FLAG = False

                for i in THUMB_JOINT:
                    mujoco_model.jnt_range[i] = freezed_joint_ranges[i]

            elif mcp >= mcp_upper and env_data.qpos[9] >= pm_upper and not MCP_FLAG:
                print('[Debug] enter mcp1')
                actions[:] = 0
                
                _freeze_joints(model = mujoco_model, data = env_data, joints_list = INDEX_JOINT, current_joints = mujoco_model.jnt_range)
                MCP_FLAG = True
            
            elif mcp <= mcp_lower and MCP_FLAG:
                print('[Debug] enter mcp2')
                actions[:] = 0
                MCP_FLAG = False

                for i in INDEX_JOINT:
                    mujoco_model.jnt_range[i] = freezed_joint_ranges[i]
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

    finally:
        STREAM.end_stream()
        env.close()

def _actuate_motors(model, command, actions, original_joint_ranges):

    actions[:] = 0

    index_joints = [7,8,9,10]
    thumb_joints = [3,4,5,6]

    if command == 'Index Contract':
        _unfreeze_joints(model = model, joints_list = index_joints, OJR = original_joint_ranges)
        actions[11], actions[15] = 0.1, 0.1                       # FPS2, FDP2   
    elif command == 'Index Release':
        _unfreeze_joints(model = model, joints_list = index_joints, OJR = original_joint_ranges)
        actions[19], actions[21] = 1.0, 1.0
        actions[28], actions[29] = 0.0, 0.0

    elif command == 'Thumb Contract':
        _unfreeze_joints(model = model, joints_list = thumb_joints, OJR = original_joint_ranges)
        actions[26], actions[24] = 0.3, 0.3

    elif command == 'Thumb Release':
        _unfreeze_joints(model = model, joints_list = thumb_joints, OJR = original_joint_ranges)
        actions[22], actions[23] = 0.3, 0.2
    else:
        actions[int(command)] = 1.0

    return actions

def _freeze_joints(model, data, joints_list, current_joints):
    for i in joints_list:
        q = model.jnt_qposadr[i]
        current = data.qpos[q]

        model.jnt_range[i] = [current, current]

    # for j in joints_list:
    #     model.jnt_range[j] = current_joints[j]

def _unfreeze_joints(model, joints_list, OJR):
    for j in joints_list:
        model.jnt_range[j] = OJR[j]

def main():
    start_simulation()

if __name__ == '__main__':
    model_folder_name = 'SingleNet_CNN+LSTM+ATTENTION_EMG_complexModel_noNorm/subject_0'
    main(model_folder_name = model_folder_name)
    

# Actuators
# idx name _
# 0 ECRL 0.0
# 1 ECRB 0.0
# 2 ECU 0.0
# 3 FCR 0.0
# 4 FCU 0.0
# 5 PL 0.0
# 6 PT 0.0
# 7 PQ 0.0
# 8 FDS5 0.0
# 9 FDS4 0.0
# 10 FDS3 0.0
# 11 FDS2 0.0
# 12 FDP5 0.0
# 13 FDP4 0.0
# 14 FDP3 0.0
# 15 FDP2 0.0
# 16 EDC5 0.0
# 17 EDC4 0.0
# 18 EDC3 0.0
# 19 EDC2 0.0
# 20 EDM 0.0
# 21 EIP 0.0
# 22 EPL 0.0            # extend thumb
# 23 EPB 0.0            # extend thumb
# 24 FPL 0.0            # bend thumb
# 25 APL 0.0            # extend thumb (maybe)
# 26 OP 0.0             # bend thumb (maybe)
# 27 RI2 0.0
# 28 LU_RB2 0.0
# 29 UI_UB2 0.0
# 30 RI3 0.0
# 31 LU_RB3 0.0
# 32 UI_UB3 0.0
# 33 RI4 0.0
# 34 LU_RB4 0.0
# 35 UI_UB4 0.0
# 36 RI5 0.0
# 37 LU_RB5 0.0
# 38 UI_UB5 0.0

# JOINTS
# idx name _
# 0 pro_sup 0.0
# 1 deviation 0.0
# 2 flexion 0.0
# 3 cmc_abduction 0.0
# 4 cmc_flexion 0.0
# 5 mp_flexion 0.0
# 6 ip_flexion 0.0
# 7 mcp2_flexion 0.0
# 8 mcp2_abduction 0.0
# 9 pm2_flexion 0.0
# 10 md2_flexion 0.0
# 11 mcp3_flexion 0.0
# 12 mcp3_abduction 0.0
# 13 pm3_flexion 0.0
# 14 md3_flexion 0.0
# 15 mcp4_flexion 0.0
# 16 mcp4_abduction 0.0
# 17 pm4_flexion 0.0
# 18 md4_flexion 0.0
# 19 mcp5_flexion 0.0
# 20 mcp5_abduction 0.0
# 21 pm5_flexion 0.0
# 22 md5_flexion 0.0