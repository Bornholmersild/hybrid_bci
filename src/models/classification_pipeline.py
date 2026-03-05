# Classification
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sherpa

# Manage datasets
import numpy as np

# Manage utils
import os 
from pathlib import Path
from datetime import datetime
import logging                  # Avoid loggings from GP
from copy import deepcopy       # Used for copy model_state
import time

# Own implementations
from src.utilities.preprocessing import EEG_preprocessing, EMG_preprocessing, RejectBadEpochs, Filtering #E402
from src.utilities.trainer_and_evaluator import FusionNet_train_eval, SingleNet_train_eval
from src.utilities.load_and_visualize_data import load_datasets

# Avoid messages for sherpa
logging.getLogger("GP").setLevel(logging.CRITICAL)
logging.getLogger("GPy").setLevel(logging.CRITICAL)

#==================#
# Global variables #
#==================#
EMG_FREQ = 2000
EEG_FREQ = 125
RMS_FREQ = 40                   # 40 for 500 samples, 125 for 32 samples (window)

EMG_LOWCUT = 20
EMG_HIGHCUT = 450
EEG_LOWCUT = 0.5
EEG_HIGHCUT = 32

EEG_NUM_CH = 3
EEG_NUM_CH = 16

TRIAL_PERIOD = 9
TRIM_PERIOD = 3

RMS_SAMPLING_WINDOW = 500           # 500 samples - 250 ms                      32 samples - 16 ms                                       
RMS_WINDOW_STEPSIZE = 50            # 50 samples - 25 ms (90 % overlap)         16 samples - 8 ms (50 % overlap)

HAMPEL_WINDOWSIZE = 100
HAMPEL_SIGMA = 2

EEG_USEABLE_CHANNELS = [2, 3, 6, 7, 10, 11]

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

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
    # Tolerance -> RANGE given [6 : 8]
    # EMG -> RANGE given by [0, 1]
    # EEG all CH -> RANGE given [0 : 3]
    # EEG 6 CH -> RANGE given [0 : 2]

#==================#
# LSTM + Attension #
#==================#
class Attension(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.attn = nn.Linear(in_features = hidden_dim, out_features = 1)
    
    def forward(self, lstm_out):
        """
        Parameter
        ---------
        lstm_out : Tensor
            Output from LSTM with shape (batch, seq_len, hidden)
        
        Return
        --------
        context : Tensor
            Weighted sum over time (batch, hidden)
        
        weights : Tensor
            attension weights (batch, seq_len, 1)
        """
        # Compute attension scores
        scores = self.attn(lstm_out)                 # (Batch, seq_len, 1)
        weights = torch.softmax(scores, dim = 1)     # Apply softmax over 'seq_len' - (Batch, seq_len, 1)

        # Weigted sum over time - NOTE: The context vector summarizes the most relevant information from the input sequence and is fed to the decoder.
        context = torch.sum(input = weights * lstm_out, dim = 1)    # (Batch, hidden)

        return context, weights
    
class SingleNet(nn.Module):
    def __init__(self, data_ch, hidden = 32, lstm_layers = 1, num_classes = 2, dropout = 0.3, activation = 'relu'):
        super().__init__()
        '''
        Args:
            input_dim - int
                The number of expected features in the input sequence at each time step (n_channels)
            hidden_dim - int
                The number of features in the hidden state. How much memory should present the hidden state at one time stamp
            layer_dim - int
                Stacking multiple LSTM layers deepens the model. If is not in the timestamp direction. But the hiddenstate goes into the input of another LSTM.
            output_dim - int
                Maps the hidden state in nn.Linear outputs to predictions (n_classes)
        '''
        first_layer = hidden // 2    # (64) // 2 = 32

        self.lstm = nn.LSTM(input_size = data_ch, 
                            hidden_size = hidden, 
                            num_layers = lstm_layers, 
                            batch_first = True)       # Input dim (batch size, sequence length, input_size)
        
        self.attension = Attension(hidden_dim = hidden)

        activations = {
            'relu' : nn.ReLU(),
            'elu' : nn.ELU()
        }
        act = activations[activation]
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden, first_layer),            # 64
            act,
            nn.Dropout(dropout),
            nn.Linear(first_layer, num_classes)
        )

    def forward(self, data):
        '''
        Runs the data through LSTM + Fully Connected layer
        Args:
            X - DataLoader tensor with dimension: (batch, seq_len, channels)
        Returns:
            out: 
                The last time step of the LSTM
        '''
        lstm_out, _ = self.lstm(data)          # lstm_out (batch, seq_len, hidden), h (1, batch, hidden)

        context, attn_weights = self.attension(lstm_out)

        logits = self.classifier(context)

        return logits, context, attn_weights

class FusionNet(nn.Module):
    def __init__(self, eeg_ch, emg_ch, hidden=32, lstm_layers = 1, num_classes=2, dropout = 0.3, activation = 'relu'):
        super().__init__()

        num_biosignals = 2
        first_layer = (hidden * num_biosignals) // 2    # (64) // 2 = 32

        self.eeg_lstm = nn.LSTM(input_size = eeg_ch, hidden_size = hidden, num_layers = lstm_layers, batch_first=True)
        self.emg_lstm = nn.LSTM(input_size = emg_ch, hidden_size = hidden, num_layers = lstm_layers, batch_first=True)

        self.attension = Attension(hidden_dim = hidden * num_biosignals)

        # WHAT OTHER ACTIVATIONS IS OF INTERST?
        activations = {
            'relu' : nn.ReLU(),
            'elu' : nn.ELU()
        }
        act = activations[activation]

        self.classifier = nn.Sequential(
            nn.Linear(hidden * num_biosignals, first_layer),            # 64
            act,
            nn.Dropout(dropout),
            nn.Linear(first_layer, num_classes)
        )

    def forward(self, eeg, emg):
        lstm_out_eeg, _ = self.eeg_lstm(eeg)          # lstm_out (batch, seq_len, hidden), h (1, batch, hidden)
        lstm_out_emg, _ = self.emg_lstm(emg)
        
        fused_seq = torch.cat([lstm_out_eeg, lstm_out_emg], dim=2)  # (batch, seq, hidden*2)
        
        context, attn_weights = self.attension(fused_seq)
        
        logits = self.classifier(context)

        return logits, context, attn_weights                        # Return dim : (batch, hidden_states for num_clases)

class SingleManageDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        '''
        Takes in the concatinated dataset of all trials, samples and channels.
        Args:
            X [ndArray] - with the dimension of (trials, samples, channels)
            y [int] - Indicate the number of trials 
        '''
        # Convert to tensors
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

        print('data shape:', self.data.shape)
        print('labels shape:', self.labels.shape)
        print()
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MultiManageDataset(torch.utils.data.Dataset):
    def __init__(self, eeg, emg, labels):
        self.eeg = torch.tensor(eeg, dtype=torch.float32)
        self.emg = torch.tensor(emg, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.rng = np.random.default_rng(SEED)

        print('eeg shape:', self.eeg.shape)
        print('emg shape:', self.emg.shape)
        print('labels shape:', self.labels.shape)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg[idx], self.emg[idx], self.labels[idx]

class Manage3Split:
    '''
    Functionality for splitting continous data into train-validation-test split and\n
    segment trial into rest, contract and release data
    '''
    def __init__(self, seed : int):
        '''
        Parameter
        ---------
        seed : int
            For np.random generator
        '''
        self.rng = np.random.default_rng(seed)

    def build_split(self, 
                    epoch_index : np.ndarray,
                    epoch_thumb : np.ndarray,
                    index_trials_indices : list,
                    thumb_trials_indices : list,
                    fs : int) -> tuple[np.ndarray, np.ndarray]:
        '''
        Provides a dataset and labels with 5 classes for the train-validation-test split

        Parameters
        ----------
        epoch_index : np.ndarray
            Epoch data for either EEG or EMG for index motion
        
        epoch_thumb : np.ndarray
            Epoch data for either EEG or EMG for thumb motion

        index_trials_indices : list
            Indices for a specfic split (index). Provided by split_trials function

        thumb_trials_indices : list
            Indices for a specfic split (thumb). Provided by split_trials function
        
        fs : int
            Sampling frequency

        Returns
        ----------
        X : np.ndarray
            Dataset split for either train, validation or test with all 5 classes
        
        y : np.ndarray
            Corresponding labels
        '''
        
        # Select trials
        index_sel = epoch_index[index_trials_indices]
        thumb_sel = epoch_thumb[thumb_trials_indices]

        # Segment
        rest_i, contract_i, release_i = self._segment_trials(index_sel, fs)
        rest_t, contract_t, release_t = self._segment_trials(thumb_sel, fs)

        # Build features
        X = np.concatenate([
            contract_i,
            release_i,
            contract_t,
            release_t,
            np.concatenate([rest_i, rest_t])
        ])

        # Build labels
        y = np.concatenate([
            np.zeros(len(contract_i)),                    # 0 index_contract
            np.ones(len(release_i)),                      # 1 index_release
            np.full(len(contract_t), 2),                  # 2 thumb_contract
            np.full(len(release_t), 3),                   # 3 thumb_release
            np.full(len(rest_i)+len(rest_t), 4)           # 4 rest
        ])

        return X, y

    def split_trials(self, num_trials : int, train_ratio : int = 0.7) -> tuple[list, list, list]:
        '''
        Provide indices for a train-validation-test split. Used per finger motion

        Parameters
        ----------
        num_trials : int
            Number of trials per finger motion

        train_ratio : int = 0.7
            Percent ratio for train split
        
        val_ratio : int = 0.15
            Percent ratio for validation and test split
        
        Returns
        ----------
        train_idx : list
            Indices for train split 
        
        val_idx : list
            Indices for validation split
        
        test_idx : list
            Indices for test split
        '''
        indices = self.rng.permutation(num_trials)

        n_train = int(train_ratio * num_trials)
        n_remain = (num_trials - n_train) // 2          # Split test and val equally 

        train_idx = indices[:n_train]
        val_idx   = indices[n_train : n_train + n_remain]
        test_idx  = indices[n_train + n_remain:]

        print(f'Train split indicies {train_idx.shape}\n',
              f'Valdiation split indicies {val_idx.shape}\n',
              f'Test split indicies {test_idx.shape}\n')

        return train_idx, val_idx, test_idx

    def _segment_trials(self, trials : np.ndarray, fs : int) -> tuple[list, list, list]:
        """
        Convert the split data into 3 classes

        Parameters
        ----------

        trials : np.ndarray
            Splited data of shape (num_trials, total_samples, channels)
        
        Returns
        ---------
        rest, contract, release : list
            Segment trial into the 3 classes
        """

        rest     = trials[:, :3*fs, :]
        contract = trials[:, 3*fs:6*fs, :]
        release  = trials[:, 6*fs:, :]

        return rest, contract, release

class ExperimentLogger:

    def __init__(self, save_path : Path):
        self.save_path = save_path / 'SHERPA_results.pt'
        
        # Load existing file if present
        if os.path.exists(save_path):
            self.results = torch.load(save_path)
        else:
            self.results = {"trials": []}

    def log_trial(
        self,
        trial_id,
        hyperparams,
        best_epoch,
        train_loss,
        val_loss,
        val_acc,
        test_loss,
        test_acc,
        preds,
        labels):
        """
        Append one trial result.
        """

        trial_result = {
            "trial_id": trial_id,
            "hyperparameters": hyperparams,
            "best_epoch": best_epoch,
            "training_loss": train_loss,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "predictions": preds,
            "labels": labels
        }

        self.results["trials"].append(trial_result)

        # Save immediately (safe against crashes)
        torch.save(self.results, self.save_path)

def load_EMG_and_EEG_data(subject_name : str | list, EEG_useable_channels : list | None = None):
    base_dir = Path().resolve().parent / 'experiment/data'

    load_ins = load_datasets(base_dir = base_dir)
    EEG_ins = EEG_preprocessing(fs = EEG_FREQ, bandpass_lowcut = EEG_LOWCUT, bandpass_highcut = EEG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    EMG_ins = EMG_preprocessing(fs = EMG_FREQ, bandpass_lowcut = EMG_LOWCUT, bandpass_highcut = EMG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    reject_ins = RejectBadEpochs(base_dir = base_dir)

    EEG_files = load_ins.find_flex_files(
        subjects = subject_name,
        modality = "EEG",
        fingers = ["index"],
        prefix = 'flex'
    )

    EMG_files = load_ins.find_flex_files(
        subjects = subject_name,
        modality = "EMG",
        fingers = ["index"],
        prefix = 'flex'
    )

    EEG, RMS, EMG, epochs_overview = load_ins.load_datasets(
        path_to_EEG_files = EEG_files,
        path_to_EMG_files = EMG_files,
        EEG_preprocessing_func = EEG_ins.preprocessing_routine,
        EMG_preprocessing_func = EMG_ins.preprocessing_routine,
        EMG_config_dict = EMG_CONFIG_DICT
    )

    # Should be in sherpa loop
    reject_mask = reject_ins.reject_routine(data_file_per_finger = EEG_files,
                                            epochs_overview = epochs_overview,
                                            EEG_data = EEG,
                                            RMS_data = RMS,
                                            reject_config_dict = REJECT_CONFIG_DICT,
                                            EEG_useable_channels = EEG_useable_channels)

    EEG = EEG[:, EEG_useable_channels].copy() if EEG_useable_channels is not None else EEG.copy()

    total_epochs = sum(epochs_overview)
    EEG_epoch = EEG.reshape(total_epochs, EEG.shape[0] // total_epochs, EEG.shape[1])
    RMS_epoch = RMS.reshape(total_epochs, RMS.shape[0] // total_epochs, RMS.shape[1])
    EMG_epoch = EMG.reshape(total_epochs, EMG.shape[0] // total_epochs, EMG.shape[1]) if EMG is not None else None

    EEG_epoch_clean = EEG_epoch[~reject_mask]
    RMS_epoch_clean = RMS_epoch[~reject_mask]
    EMG_epoch_clean = EMG_epoch[~reject_mask] if EMG is not None else None

    filt_ins = Filtering()
    EEG_car = EEG_epoch_clean - np.mean(EEG_epoch_clean, axis = 2, keepdims = True)
    # EEG_epoch_norm = filt_ins.zscore(EEG_car, mode = 'within_ch')
    RMS_epoch_norm = filt_ins.zscore(RMS_epoch_clean, mode = 'within_ch')
    # EMG_epoch_norm = filt_ins.zscore(EMG_epoch_clean, mode = 'within_ch')
    # Add car
    # Add normalazation

def load_EEG_data(subject_name : str | list, finger_name : str):
    base_dir = Path(__file__).resolve().parents[2] / 'src/experiment/data'

    load_ins = load_datasets(base_dir = base_dir)
    EEG_ins = EEG_preprocessing(fs = EEG_FREQ, bandpass_lowcut = EEG_LOWCUT, bandpass_highcut = EEG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    reject_ins = RejectBadEpochs(base_dir = base_dir)

    EEG_files = load_ins.find_flex_files(
        subjects = subject_name,
        modality = "EEG",
        fingers = finger_name,
        prefix = 'flex'
    )

    EEG, epochs_overview = load_ins.load_datasets_EEG(
        path_to_data_files = EEG_files,
        preprocessing_func = EEG_ins.preprocessing_routine,
    )

    # Should be in sherpa loop 
    reject_mask = reject_ins.reject_routine(data_file_per_finger = EEG_files,
                                            epochs_overview = epochs_overview,
                                            EEG_data = EEG,
                                            RMS_data = None,
                                            reject_config_dict = REJECT_CONFIG_DICT,
                                            EEG_useable_channels = None)

    total_epochs = sum(epochs_overview)
    EEG_epoch = EEG.reshape(total_epochs, EEG.shape[0] // total_epochs, EEG.shape[1])

    EEG_epoch_clean = EEG_epoch[~reject_mask]

    EEG_car = EEG_epoch_clean - np.mean(EEG_epoch_clean, axis = 2, keepdims = True)

    filt_ins = Filtering()
    EEG_epoch_norm = filt_ins.zscore(EEG_car, mode = 'within_ch')

    return EEG_epoch_norm, epochs_overview

def load_EMG_data(subject_name : str | list, finger_name : str):
    base_dir = Path(__file__).resolve().parents[2] / 'src/experiment/data'

    load_ins = load_datasets(base_dir = base_dir)
    EMG_ins = EMG_preprocessing(fs = EMG_FREQ, bandpass_lowcut = EMG_LOWCUT, bandpass_highcut = EMG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    reject_ins = RejectBadEpochs(base_dir = base_dir)

    EMG_files = load_ins.find_flex_files(
        subjects = subject_name,
        modality = "EMG",
        fingers = finger_name,
        prefix = 'flex'
    )

    RMS, EMG, epochs_overview = load_ins.load_datasets_EMG(
        path_to_data_files = EMG_files,
        preprocessing_func = EMG_ins.preprocessing_routine,
        EMG_config_dict = EMG_CONFIG_DICT
    )

    # Should be in sherpa loop 
    reject_mask = reject_ins.reject_routine(data_file_per_finger = EMG_files,
                                            epochs_overview = epochs_overview,
                                            EEG_data = None,
                                            RMS_data = RMS,
                                            reject_config_dict = REJECT_CONFIG_DICT,
                                            EEG_useable_channels = None)

    total_epochs = sum(epochs_overview)
    RMS_epoch = RMS.reshape(total_epochs, RMS.shape[0] // total_epochs, RMS.shape[1])
    EMG_epoch = EMG.reshape(total_epochs, EMG.shape[0] // total_epochs, EMG.shape[1]) if EMG is not None else None

    RMS_epoch_clean = RMS_epoch[~reject_mask]
    EMG_epoch_clean = EMG_epoch[~reject_mask] if EMG is not None else None

    filt_ins = Filtering()
    RMS_epoch_norm = filt_ins.zscore(RMS_epoch_clean, mode = 'within_ch')
    EMG_epoch_norm = filt_ins.zscore(EMG_epoch_clean, mode = 'within_ch') if EMG is not None else None

    return RMS_epoch_norm, EMG_epoch_norm

def build_optimizer(model_params, trial_parameters):
    hp = trial_parameters
    opt_name = hp["optimizer"]
    lr = float(hp["learning_rate"])
    wd = float(hp["weight_decay"])

    if opt_name == "adamw":
        # AdamW: decoupled weight decay
        return torch.optim.AdamW(model_params, lr=lr, weight_decay=wd)

    elif opt_name == "sgd_momentum":
        mom = float(hp["momentum"])
        nes = bool(hp["nesterov"])
        # SGD: in PyTorch this is classic L2-style weight decay, which is fine/equivalent for SGD
        return torch.optim.SGD(model_params, lr=lr, momentum=mom, nesterov=nes, weight_decay=wd)

    raise ValueError(f"Unknown optimizer: {opt_name}")
    
def load_classfication(subject_name : str | list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()              # Use pin_memory if CUDA is available
    print(f"Using device: {device}")
    print("Pin memory set to:", pin_memory)

    LOG_NAME = f'{subject_name}_SingleNet_EMG'
    log_dir = Path(__file__).resolve().parent / f'loggings/{LOG_NAME}'         # Path(__file__).resolve() -> Absolute path to this file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    logger = ExperimentLogger(save_path = log_dir)

    #===========#
    # Load data #
    #===========#
    X_epoch_index, _ = load_EMG_data(subject_name = subject_name, finger_name = 'index')
    X_epoch_thumb, _ = load_EMG_data(subject_name = subject_name, finger_name = 'thumb')
    FREQ = RMS_FREQ
    print(X_epoch_index.shape, X_epoch_thumb.shape)

    split_ins = Manage3Split(seed = SEED)

    num_index_trials = X_epoch_index.shape[0]
    num_thumb_trials = X_epoch_thumb.shape[0]

    train_i, val_i, test_i = split_ins.split_trials(num_index_trials)
    train_t, val_t, test_t = split_ins.split_trials(num_thumb_trials)

    X_train, y_train = split_ins.build_split(epoch_index = X_epoch_index,
                                            epoch_thumb = X_epoch_thumb,
                                            index_trials_indices = train_i,
                                            thumb_trials_indices = train_t,
                                            fs = FREQ)
    X_val, y_val = split_ins.build_split(epoch_index = X_epoch_index,
                                            epoch_thumb = X_epoch_thumb,
                                            index_trials_indices = val_i,
                                            thumb_trials_indices = val_t,
                                            fs = FREQ)
    X_test, y_test = split_ins.build_split(epoch_index = X_epoch_index,
                                            epoch_thumb = X_epoch_thumb,
                                            index_trials_indices = test_i,
                                            thumb_trials_indices = test_t,
                                            fs = FREQ)
    

    #=======================#
    # Multi fusion datasets #
    #=======================#
    # ...

    #=================#
    # Single datasets #
    #=================#
    train_eval_ins = SingleNet_train_eval()

    print('Training dataset shapes:')
    train_dataset_ins = SingleManageDataset(X_train, y_train)
    print('Validation dataset shapes:')
    val_dataset_ins = SingleManageDataset(X_val, y_val)
    print('Testing dataset shapes:')
    test_dataset_ins = SingleManageDataset(X_test, y_test)

    #========================================================#
    # THESE PARAMETERS ARE CHANCEABLE, DEPENDING ON THE TASK #
    #========================================================#
    MAX_NUM_TRIALS = 100             # 75 - 250 (simply to max) 
    DATA_CH = X_epoch_index.shape[2]
    NUM_CLASSES = 5
    NUM_EPOCHS = 150                 # 150 - 200
    PATIENCE = 25                   # Early stopping patience - 25
    WHICH_NETWORK = 'SingleNet'     # SingleNet or FusionNet -> Used to adjust model_args for torch.save model

    # Used with FusionNet
    EEG_CH = 0
    EMG_CH = 0

    assert str.lower(WHICH_NETWORK) in ["singlenet", "fusionnet"], \
    f"Invalid WHICH_NETWORK: {WHICH_NETWORK}"

    #====================================#
    # SHERPA Hyperparameter Optimazation #
    #====================================#

    parameters = [sherpa.Continuous(name='learning_rate', range=[0.00001, 0.001], scale='log'),
              sherpa.Continuous(name='dropout', range=[0.1, 0.5]),
              sherpa.Ordinal(name='batch_size', range=[16, 32, 64]),
              sherpa.Discrete(name='num_hidden_units', range=[32, 64]),         # before 256
              sherpa.Choice(name='activation', range=['relu', 'elu']),
              #sherpa.Ordinal(name='lstm_layers', range=[1, 3]),
              sherpa.Choice(name="optimizer", range=["adamw", "sgd_momentum"]),
              sherpa.Continuous(name="weight_decay", range=[1e-6, 1e-2], scale="log"),
              sherpa.Continuous(name="momentum", range=[0.7, 0.99]),   # only used for SGD
              sherpa.Choice(name="nesterov", range=[False, True]),     # only used for SGD])
    ]
    
    # algorithm = sherpa.algorithms.RandomSearch(max_num_trials = MAX_NUM_TRIALS)
    algorithm = sherpa.algorithms.GPyOpt(
        max_num_trials = MAX_NUM_TRIALS,
        acquisition_type = 'EI',                     # Expected improvement
        num_initial_data_points = 20                 # Number of hyperparameter configurations before model learns
    )
    # Study represents the hyperparameter optimization itself
    study = sherpa.Study(
        parameters = parameters,
        algorithm = algorithm,
        lower_is_better = True,
        disable_dashboard = True
    )

    for trial in study:
        dropout = trial.parameters['dropout']       
        batch_size = trial.parameters['batch_size']
        num_hidden_units = trial.parameters['num_hidden_units']
        activation = trial.parameters['activation'] 
        lstm_layers = 1 #trial.parameters['lstm_layers']     # trial.parameters['lstm_layers']

        #=======================#
        # Multi fusion datasets #
        #=======================#
        # model = FusionNet(eeg_ch = 6, emg_ch = 3, hidden = num_hidden_units, lstm_layers = 1, num_classes = NUM_CLASSES, dropout = dropout, activation = activation)
        
        #=================#
        # Single datasets #
        #=================#
        model = SingleNet(data_ch = DATA_CH, hidden = num_hidden_units, lstm_layers = lstm_layers, num_classes = NUM_CLASSES, dropout = dropout, activation = activation)

        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(model_params = model.parameters(), trial_parameters = trial.parameters)

        # DataLoaders (update batch_size)
        train_loader = DataLoader(train_dataset_ins, batch_size = batch_size, shuffle = True, pin_memory = pin_memory, num_workers = 0)
        val_loader = DataLoader(val_dataset_ins, batch_size = batch_size, shuffle = False, pin_memory = pin_memory, num_workers = 0)
        test_loader = DataLoader(test_dataset_ins, batch_size = batch_size, shuffle = False, pin_memory = pin_memory, num_workers = 0)

        best_train_loss = None
        best_val_loss = float("inf")
        best_val_acc = None

        best_epoch = 0
        best_state_dict = None
        early_stopping_counter = 0

        # Create log folder
        log_folder = os.path.join(log_dir, f"trial_{trial.id}")
        os.makedirs(log_folder, exist_ok=False)
        writer = SummaryWriter(os.path.join(log_folder, 'trial_{}_timestamp_{}'.format(trial.id, timestamp)))

        for epoch in range(NUM_EPOCHS):

            # Train model
            avg_train_loss = train_eval_ins.train_one_epoch(model = model, train_loader = train_loader, criterion = criterion, optimizer = optimizer, device = device)

            # Validate model
            avg_vloss, vacc, _ = train_eval_ins.validaton_one_epoch(model = model, val_loader = val_loader, criterion = criterion, device = device)

            # Tensor Board logging
            writer.add_scalars('Loss', { 'Training' : avg_train_loss, 'Validation' : avg_vloss }, epoch + 1)
            writer.add_scalars('Accuracy Validation', {'Validation' : vacc }, epoch + 1)
            writer.flush()

            study.add_observation(trial = trial,
                                iteration = epoch,
                                objective = avg_vloss)

            # Track best performance, and save the model's state
            if avg_vloss < best_val_loss:
                best_val_loss = avg_vloss
                best_epoch = epoch

                best_state_dict = deepcopy(model.state_dict())
                best_optimizer_dict = deepcopy(optimizer.state_dict())
 
                best_train_loss = avg_train_loss
                best_val_acc = vacc

                early_stopping_counter = 0

            else:
                early_stopping_counter += 1

                if early_stopping_counter >= PATIENCE:
                    break

            print(
                f'{subject_name} | '
                f'Trial {trial.id}/{MAX_NUM_TRIALS} | '
                f'Epoch {epoch+1}/{NUM_EPOCHS} | '
                f'Train {avg_train_loss:.4f} | '
                f'Val {avg_vloss:.4f} | '
                f'Acc {vacc:.2f} |',
                f'Early stopping {early_stopping_counter}',
                end='\r',
                flush=True
            )

        model.load_state_dict(best_state_dict)

        avg_test_loss, test_acc, predictions, labels = train_eval_ins.inference_one_epoch(model = model, test_loader = test_loader, criterion = criterion, device = device)

        if str.lower(WHICH_NETWORK) == 'singlenet':
            model_arg = {
                "data_ch": DATA_CH,
                "hidden": num_hidden_units,
                "lstm_layers": lstm_layers,
                "num_classes": NUM_CLASSES,
                "dropout": dropout,
                "activation": activation,}
        elif str.lower(WHICH_NETWORK) == 'fusionnet':
            model_arg = {
                "eeg_ch": EEG_CH,
                "emg_ch": EMG_CH,
                "hidden": num_hidden_units,
                "lstm_layers": lstm_layers,
                "num_classes": NUM_CLASSES,
                "dropout": dropout,
                "activation": activation,}
            
        torch.save({
            "model_state": best_state_dict,
            "model_args": model_arg, 
            "optimizer_state_dict": best_optimizer_dict,
            "hyperparameters": trial.parameters,}, 
            r'{}\model.pth'.format(log_folder))
        
        logger.log_trial(
            trial_id=trial.id,
            hyperparams = trial.parameters,
            best_epoch = best_epoch,
            train_loss = best_train_loss,
            val_loss = best_val_loss,
            val_acc = best_val_acc,
            test_loss = avg_test_loss,
            test_acc = test_acc,
            preds = predictions,
            labels = labels)

        writer.close()
        study.finalize(trial, status = 'COMPLETED')

def main():
    t0 = time.time()
    subjects = ['subject_12']

    for subject in subjects:
        load_classfication(subject_name = subject)

    print('Classification COMPLETE\n'
          'Time it took: ', time.time() - t0, 's')

def inspect_model(logging_name = 'SingleNet_EMG/subject_11_SingleNet_EMG'):
    sherpa_info_path = Path(__file__).resolve().parent / f"loggings/{logging_name}/SHERPA_results.pt"

    if not os.path.exists(sherpa_info_path):
        raise FileExistsError(sherpa_info_path)
    data = torch.load(sherpa_info_path, weights_only=False)

    # print(data['trials'][57])
    acc_list = []
    for acc in data['trials']:
        acc_list.append(acc['test_accuracy'])
    
    acc_list.sort(reverse=True)
    for i, acc in enumerate(acc_list):
        print(i, acc)
    
    best = min(
        data["trials"],
        key=lambda x: x["validation_loss"]
    )
    # print(best)
    print()
    # print(data['trials'][39])
    print()
    print('Best trial ID: ', best['trial_id'])
    print('Stoped at epoch', best['best_epoch'])
    print('Training loss:' , best['training_loss'])
    print('validation loss', best['validation_loss'])
    print('Test accuracy: ', best["test_accuracy"])
    print('Hyperparameter: ', best["hyperparameters"])

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(best['labels'], best['predictions']) 

    print(cm)


    # Load model
    # model_path = Path(__file__).resolve().parent / f"loggings/{logging_name}/trial_{best['trial_id']}/model.pth"
    # checkpoint = torch.load(model_path, map_location="cpu")

    # model_args = checkpoint["model_args"]

    # model = SingleNet(**model_args)
    # model.load_state_dict(checkpoint["model_state"])
    # model.eval()

    # dummy_input = torch.randn(1, 120, 3)  # (batch, samples, channels)
    # with torch.no_grad():
    #     output, _, _ = model(dummy_input)

    # print("Output shape:", output.shape)

    # for name, param in model.named_parameters():
    #     print(name, param.mean().item())
    #     break

if __name__ == '__main__':
    main()
    # inspect_model()