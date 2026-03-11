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
import matplotlib.pyplot as plt

# Analysis
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

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
    
class SingleNet_LSTM_ATTENSION(nn.Module):
    def __init__(self, input_dim : np.ndarray, output_dim : int, hidden_dim : int, lstm_layers : int, bidirectional : bool, dropout : float, activation : str, dense_ratio : int):
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
        self.bidirectional = bidirectional

        if bidirectional:                       # For Bi-LSTM : hidden units is doubled
            dense_hidden_layers = hidden_dim * 2
        else:
            dense_hidden_layers = hidden_dim

        dense_layers = max(8, int(dense_hidden_layers * dense_ratio))

        self.lstm = nn.LSTM(input_size = input_dim, 
                            hidden_size = hidden_dim, 
                            num_layers = lstm_layers, 
                            batch_first = True)       # Input dim (batch size, sequence length, input_size)
        
        self.attension = Attension(hidden_dim = dense_hidden_layers)

        activations = {
            'relu' : nn.ReLU(),
            'elu' : nn.ELU()
        }
        act = activations[activation]
        
        self.classifier = nn.Sequential(
            nn.Linear(dense_hidden_layers, dense_layers),            # 64
            act,
            nn.Dropout(dropout),
            nn.Linear(dense_layers, output_dim)
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

class SingleNet_LSTM(nn.Module):
    '''
    Single network to perform LSTM on EEG or EMG datasets.\n
    Returns logits from LSTM features to number of classes to classify

    Parameters
    ----------
    input_dim : int
        The number of expected features in the input sequence at each time step (n_channels)
    output_dim : int
        Maps the hidden state in nn.Linear outputs to predictions (n_classes)
    hidden_dim : int (Optional hyperparameter)
        The number of features in the hidden state. How much memory should present the hidden state at one time stamp
    lstm_layers : int (Optional hyperparameter)
        Stacking multiple LSTM layers deepens the model. If is not in the timestamp direction. But the hiddenstate goes into the input of another LSTM.
    bidirectional : int (Optional hyperparameter)
        Enable bi-directional LSTM. Doubles the hidden_dim dimensionality.
    dropout : float (Optional hyperparameter)
        Introduce a dropout probability on the outputs of each LSTM layers. Except for the final layer.
    dense_ratio : int (Optional hyperparameter)
        Embed higher-dimensional space (dense_hidden_layers < dense_layers) -> learns more complex nonlienar combinations -> risk of overfitting.
        Distill LSTM features into a compact representation (dense_hidden_layers > dense_layers) -> strong regularization -> risk of underfit and removal of important patterns
    '''
    def __init__(self, input_dim : np.ndarray, output_dim : int, hidden_dim : int, lstm_layers : int, bidirectional : bool, dropout : float, activation : str, dense_ratio : int):
        super().__init__()
        self.bidirectional = bidirectional

        if bidirectional:                       # For Bi-LSTM : hidden units is doubled
            dense_hidden_layers = hidden_dim * 2
        else:
            dense_hidden_layers = hidden_dim
        
        dense_layers = max(8, int(dense_hidden_layers * dense_ratio))

        self.lstm = nn.LSTM(input_size = input_dim, 
                            hidden_size = hidden_dim,            # Don't consider bidirectional for hidden units. 
                            num_layers = lstm_layers, 
                            batch_first = True,                  # Data input will be (batch, seq_len, channels)
                            dropout = dropout,
                            bidirectional = bidirectional)       # Input dim (batch size, sequence length, input_size)

        activations = {
            'relu' : nn.ReLU(),
            'elu' : nn.ELU()
        }
        act = activations[activation]
        
        self.classifier = nn.Sequential(
            nn.Linear(dense_hidden_layers, dense_layers),            
            act,
            nn.Dropout(dropout),
            nn.Linear(dense_layers, output_dim)
        )

    def forward(self, data : torch.Tensor) -> tuple[torch.Tensor, None, None]:
        '''
        Runs the data through LSTM + Fully Connected layer

        Parameters
        ----------
        data : torch.Tensor
            DataLoader tensor with dimension: (batch, seq_len, channels) for either EEG or EMG
        
        Returns
        ----------
        logits : torch.Tensor
            Linear transform of LSTM features -> logits
        _ : None
            Placeholder
        _ : None
            Placeholder
        '''
        _, (hn, _) = self.lstm(data)          # Final LSTM hidden layer : hn (bi * num_layers, batch, hidden) - bi = 2 if bidirectional == True

        if self.bidirectional:                # Last two elements contain final forward and final reverse hidden states
            h_forward = hn[-2, :, :]          # (1, batch, hidden)
            h_backward = hn[-1, :, :]         # (1, batch, hidden)
            h_final = torch.cat((h_forward, h_backward), dim = 1)       # Concat -> (batch, hidden * 2)
        else:
            h_final = hn[-1]                  # (batch, hidden)

        logits = self.classifier(h_final)

        return logits, None, None

class SingleNet_CNN(nn.Module):
    def __init__(self, data_ch, num_classes = 2, dropout = 0.3, activation = 'relu', cnn_filters = 32, kernel_size = 5):
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

        activations = {
            'relu' : nn.ReLU(),
            'elu' : nn.ELU()
        }
        act = activations[activation]

        # ---- CNN MODULE ----
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels = data_ch, out_channels = cnn_filters, kernel_size = kernel_size, padding = kernel_size // 2),        
            nn.BatchNorm1d(num_features = cnn_filters),
            act,
            nn.AvgPool1d(kernel_size = 2),

            nn.Conv1d(in_channels = cnn_filters, out_channels = cnn_filters*2, kernel_size = kernel_size, padding = kernel_size // 2),        
            nn.BatchNorm1d(num_features = cnn_filters*2),
            act,           
            nn.AvgPool1d(kernel_size = 2),
            
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(cnn_filters*2, cnn_filters),            # 64
            act,
            nn.Dropout(dropout),
            nn.Linear(cnn_filters, num_classes)
        )
    
    def forward(self, data):
        # data: (batch, seq_len, channels)

        x = data.permute(0, 2, 1)        # (batch, channels, seq_len)
        
        x = self.cnn(x)                  # (batch, cnn_filters, seq_len/2)
        
        # Concatenate channel embeddings
        x = torch.mean(x, dim=2)   # (B, F)

        logits = self.classifier(x)

        return logits, None, None        # None is placeholders

class SingleNet_CNN_LSTM_ATTENSION(nn.Module):
    def __init__(self, data_ch, num_classes = 2, dropout = 0.3, activation = 'relu', lstm_layers = 1, hidden = 64, cnn_filters = 32, kernel_size = 5):
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

        activations = {
            'relu' : nn.ReLU(),
            'elu' : nn.ELU()
        }
        act = activations[activation]

        # ---- CNN MODULE ----
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels = data_ch, out_channels = cnn_filters, kernel_size = kernel_size, padding = kernel_size // 2),        
            nn.BatchNorm1d(num_features = cnn_filters),
            act,
            nn.AvgPool1d(kernel_size = 2),

            nn.Conv1d(in_channels = cnn_filters, out_channels = cnn_filters*2, kernel_size = kernel_size, padding = kernel_size // 2),        
            nn.BatchNorm1d(num_features = cnn_filters*2),
            act,           
            nn.AvgPool1d(kernel_size = 2),
            
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size = cnn_filters*2,
            hidden_size = hidden,
            num_layers = lstm_layers,
            batch_first = True
        )

        self.attension = Attension(hidden_dim = hidden)

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),            # 64
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes)
        )
    
    def forward(self, data):
        # data: (batch, seq_len, channels)

        x = data.permute(0, 2, 1)        # (batch, channels, seq_len)
        x = self.cnn(x)                  # (batch, cnn_filters, seq_len/2)
        x = x.permute(0, 2, 1)           # (batch, seq_len/2, cnn_filters)
        
        lstm_out, _ = self.lstm(x)       # (batch, seq_len, hidden)

        context, attn_weights = self.attension(lstm_out)

        logits = self.classifier(context)

        return logits, context, attn_weights        # None is placeholders

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
        
        # X = np.concatenate([
        #     contract_t,
        #     rest_t
        # ])

        # # Build labels
        # y = np.concatenate([
        #     np.zeros(len(contract_t)),                    # 0 index_contract
        #     np.ones(len(rest_t))
        # ])
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

    LOG_NAME = f'{subject_name}'
    log_dir = Path(__file__).resolve().parent / f'loggings/SingleNet_LSTM_EMG_3GO/{LOG_NAME}'         # Path(__file__).resolve() -> Absolute path to this file
    data_dir = Path(__file__).resolve().parents[2] / 'src/experiment/data'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    logger_ins = ExperimentLogger(save_path = log_dir)
    load_ins = load_datasets(base_dir = data_dir)
    split_ins = Manage3Split(seed = SEED)
    EMG_ins = EMG_preprocessing(fs = EMG_FREQ, bandpass_lowcut = EMG_LOWCUT, bandpass_highcut = EMG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    # EEG_ins = EEG_preprocessing(fs = EEG_FREQ, bandpass_lowcut = EEG_LOWCUT, bandpass_highcut = EEG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)

    #===========#
    # Load data #
    #===========#
    X_epoch_index, _, _ = load_ins.load_EMG_data(subject_name = subject_name, finger_name = 'index', EMG_config_dict = EMG_CONFIG_DICT, reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EMG_ins.preprocessing_routine)
    X_epoch_thumb, _, _ = load_ins.load_EMG_data(subject_name = subject_name, finger_name = 'thumb', EMG_config_dict = EMG_CONFIG_DICT, reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EMG_ins.preprocessing_routine)

    FREQ = RMS_FREQ

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

    print('\nTraining dataset shapes:')
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

    parameters = [
              sherpa.Continuous(name='learning_rate', range=[0.00001, 0.001], scale='log'),
              sherpa.Continuous(name='dropout', range=[0.1, 0.5]),
              sherpa.Ordinal(name='batch_size', range=[16, 32, 64]),
              sherpa.Ordinal(name='dense_ratio', range=[0.25, 0.5, 0.7, 1.0]),
              sherpa.Choice(name='activation', range=['relu', 'elu']),
              sherpa.Choice(name="optimizer", range=["adamw", "sgd_momentum"]),
              sherpa.Continuous(name="weight_decay", range=[1e-6, 1e-2], scale="log"),
              sherpa.Continuous(name="momentum", range=[0.7, 0.99]),                    # only used for SGD
              sherpa.Choice(name="nesterov", range=[False, True]),                      # only used for SGD])
              #sherpa.Ordinal(name='lstm_layers', range=[1]),
              sherpa.Discrete(name='num_hidden_units', range=[32, 64]),         # before 256
              #sherpa.Choice(name="bidirectional", range=[False, True])                      # only used for SGD])
    ]
    
    # algorithm = sherpa.algorithms.RandomSearch(max_num_trials = MAX_NUM_TRIALS)
    algorithm = sherpa.algorithms.GPyOpt(
        max_num_trials = MAX_NUM_TRIALS,
        acquisition_type = 'EI',                     # Expected improvement
        num_initial_data_points = 50                 # Number of hyperparameter configurations before model learns
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
        dense_ratio = trial.parameters['dense_ratio']
        lstm_layers = 1                                   
        bidirectional = False                             

        ''' Optinal hyperparameters
        trial.parameters['lstm_layers']
        trial.parameters['bidirectional']
        '''
        #=======================#
        # Multi fusion datasets #
        #=======================#
        # model = FusionNet(eeg_ch = 6, emg_ch = 3, hidden = num_hidden_units, lstm_layers = 1, num_classes = NUM_CLASSES, dropout = dropout, activation = activation)
        
        #=================#
        # Single datasets #
        #=================#
        model = SingleNet_LSTM_ATTENSION(input_dim = DATA_CH, output_dim = NUM_CLASSES, hidden_dim = num_hidden_units, lstm_layers = lstm_layers, bidirectional = bidirectional, dropout = dropout, activation = activation, dense_ratio = dense_ratio)

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
                f'Early stopping {early_stopping_counter} |',
                end='\r',
                flush=True
            )

        model.load_state_dict(best_state_dict)

        avg_test_loss, test_acc, predictions, labels = train_eval_ins.inference_one_epoch(model = model, test_loader = test_loader, criterion = criterion, device = device)

        if str.lower(WHICH_NETWORK) == 'singlenet':
            model_arg = {
                "input_dim": DATA_CH,
                "output_dim" : NUM_CLASSES,
                "hidden_dim": num_hidden_units,
                "lstm_layers": lstm_layers,
                "bidirectional" : bidirectional,
                "dropout": dropout,
                "activation": activation,
                "dense_ratio": dense_ratio,}
        elif str.lower(WHICH_NETWORK) == 'fusionnet':
            model_arg = {
                "eeg_dim": EEG_CH,
                "emg_dim": EMG_CH,
                "output_dim" : NUM_CLASSES,
                "hidden_dim": num_hidden_units,
                "lstm_layers": lstm_layers,
                "bidirectional" : bidirectional,
                "dropout": dropout,
                "activation": activation,
                "dense_ratio": dense_ratio,}
            
        torch.save({
            "model_state": best_state_dict,
            "model_args": model_arg, 
            "optimizer_state_dict": best_optimizer_dict,
            "hyperparameters": trial.parameters,}, 
            r'{}\model.pth'.format(log_folder))
        
        logger_ins.log_trial(
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

def inspect_model(logging_name = 'SingleNet_EMG/subject_11_SingleNet_EMG'):
    # sherpa_info_path = Path(__file__).resolve().parent / f"loggings/{logging_name}/SHERPA_results.pt"

    inspect_list = [0]
    acc_list = []
    for i in inspect_list:
            
        sherpa_info_path = Path(__file__).resolve().parent / f"loggings/SingleNet_LSTM_EMG_3GO/subject_{i}/SHERPA_results.pt"
        if not os.path.exists(sherpa_info_path):
            raise FileExistsError(sherpa_info_path)
        data = torch.load(sherpa_info_path, weights_only=False)

        # print(data['trials'][57])
        acc_list = []
        for acc in data['trials']:
            acc_list.append(acc['test_accuracy'])
            print(acc['test_accuracy'])
            print(acc['hyperparameters'], '\n')
        
        acc_list.sort(reverse=True)
        for i, acc in enumerate(acc_list):
            # print(i, acc)
            pass
        
        best = max(
            data["trials"],
            key=lambda x: x["test_accuracy"]
        )
        print(f'\n---------Subject {i}-----------')
        print('Best trial ID: ', best['trial_id'])
        print('Stoped at epoch', best['best_epoch'])
        print('Training loss:' , best['training_loss'])
        print('validation loss', best['validation_loss'])
        print('Test accuracy: ', best["test_accuracy"])
        # print('Hyperparameter: ', best["hyperparameters"])

        acc_list.append(best["test_accuracy"])

        cm = confusion_matrix(best['labels'], best['predictions']) 
        print(cm)

    print('\n---------MEAN and STD-----------')
    print('Mean across subjects :', np.mean(acc_list))
    print('Std across subjects :', np.std(acc_list))

def inspect_model_SNE(subject_name = 'subject_0', sherpa_log_folder = 'SingleNet_LSTM_EMG'):
    model_path_folder = Path(__file__).resolve().parent / f"loggings/{sherpa_log_folder}/{subject_name}"
    sherpa_info_path = model_path_folder / 'SHERPA_results.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()              # Use pin_memory if CUDA is available

    if not os.path.exists(sherpa_info_path):
        raise FileExistsError(sherpa_info_path)
    
    data = torch.load(sherpa_info_path, weights_only=False)
        
    best = min(
        data["trials"],
        key=lambda x: x["validation_loss"]
    )

    best_trial_id = best['trial_id']
    batch_size = best['hyperparameters']['batch_size']
    print(f'\n---------Subject {0}-----------')
    print('Best trial ID: ', best_trial_id)
    print('Stoped at epoch', best['best_epoch'])
    print('Training loss:' , best['training_loss'])
    print('validation loss', best['validation_loss'])
    print('Test accuracy: ', best["test_accuracy"])
    print('\n')
    
    model_path = model_path_folder / f'trial_{best_trial_id}/model.pth'
    checkpoint = torch.load(f = model_path, map_location = device)

    model_interference = SingleNet_LSTM(**checkpoint["model_args"])
    model_interference.load_state_dict(checkpoint["model_state"])
    model_interference.to(device)

    #===================#
    # Load Test dataset #
    #===================#
    data_dir = Path(__file__).resolve().parents[2] / 'src/experiment/data'
    
    load_ins = load_datasets(base_dir = data_dir)
    split_ins = Manage3Split(seed = SEED)
    EMG_ins = EMG_preprocessing(fs = EMG_FREQ, bandpass_lowcut = EMG_LOWCUT, bandpass_highcut = EMG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    # EEG_ins = EEG_preprocessing(fs = EEG_FREQ, bandpass_lowcut = EEG_LOWCUT, bandpass_highcut = EEG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)

    X_epoch_index, _, _ = load_ins.load_EMG_data(subject_name = subject_name, finger_name = 'index', EMG_config_dict = EMG_CONFIG_DICT, reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EMG_ins.preprocessing_routine)
    X_epoch_thumb, _, _ = load_ins.load_EMG_data(subject_name = subject_name, finger_name = 'thumb', EMG_config_dict = EMG_CONFIG_DICT, reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EMG_ins.preprocessing_routine)

    FREQ = RMS_FREQ

    num_index_trials = X_epoch_index.shape[0]
    num_thumb_trials = X_epoch_thumb.shape[0]

    _, _, test_i = split_ins.split_trials(num_index_trials)
    _, _, test_t = split_ins.split_trials(num_thumb_trials)

    X_test, y_test = split_ins.build_split(epoch_index = X_epoch_index,
                                            epoch_thumb = X_epoch_thumb,
                                            index_trials_indices = test_i,
                                            thumb_trials_indices = test_t,
                                            fs = FREQ)
    print('Test 1 segment : ', X_test[0, 0:10, 2])
    print('Test 2 segment : ', X_test[7, 30:40, 1])
    print('Test 3 segment : ', X_test[17, 90:100, 0])
    
    #=================#
    # Single datasets #
    #=================#
    print('Testing dataset shapes:')
    test_dataset_ins = SingleManageDataset(X_test, y_test)

    test_loader = DataLoader(test_dataset_ins, batch_size = batch_size, shuffle = False, pin_memory = pin_memory, num_workers = 0)

    #===================#
    # Perform inference #
    #===================#
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_logits = []
    # all_context = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits, _, _ = model_interference(inputs)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_logits.append(logits.cpu())
            # all_context.append(context.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_logits = torch.cat(all_logits).numpy()
        # all_context = torch.cat(all_context).numpy()

        #==========#
        # Analysis #
        #==========#

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)

        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        print(cm_norm)

        accuracy = correct / total
        print(f"Test accuracy: {accuracy:.4f}")

    plot_tsne_context(context = all_logits, labels = y_test, perplexity = 40, n_iter = 1000, random_state = 42)
    # plot_tsne_context(context = all_context, labels = y_test, perplexity = 40, n_iter = 1000, random_state = 42)

    score = silhouette_score(all_logits, y_test)
    print(score)

    # ~0.5 → good separation
    # ~0.2 → weak separation
    # ~0 → no separation
    # <0 → overlapping

def plot_tsne_context(
    context,
    labels,
    perplexity=30,
    n_iter=1000,
    random_state=42,
    title="t-SNE of Attention Context",
    figsize=(8, 6)
):
    """
    Plot t-SNE embedding of attention context vectors.

    Args:
        context (torch.Tensor or np.ndarray):
            Shape (n_samples, feature_dim)
        labels (torch.Tensor or np.ndarray):
            Shape (n_samples,)
        perplexity (int):
            t-SNE perplexity (typ. 5-50)
        n_iter (int):
            Number of optimization iterations
        random_state (int):
            Seed for reproducibility
        title (str):
            Plot title
        figsize (tuple):
            Figure size
    """

    # ---- detach safely ----
    if hasattr(context, "detach"):
        X = context.detach().cpu().numpy()
    else:
        X = np.asarray(context)

    if hasattr(labels, "detach"):
        y = labels.detach().cpu().numpy()
    else:
        y = np.asarray(labels)

    # ---- t-SNE ----
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        init="pca",
        learning_rate="auto",
        random_state=random_state
    )

    X_embedded = tsne.fit_transform(X)

    # ---- plot ----
    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=y,
        cmap="tab10",
        alpha=0.7,
        s=25
    )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(title)
    plt.colorbar(scatter, label="Class")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    t0 = time.time()
    subjects = ['subject_0']

    for subject in subjects:
        load_classfication(subject_name = subject)

    print('Classification COMPLETE\n'
          'Time it took: ', time.time() - t0, 's')


if __name__ == '__main__':
    
    # main()
    inspect_model_SNE('subject_0')

    '''subject_name = 'subject_0'
    LOG_NAME = f'{subject_name}'
    log_dir = Path(__file__).resolve().parent / f'loggings/SingleNet_LSTM_EMG_2GO/{LOG_NAME}'         # Path(__file__).resolve() -> Absolute path to this file
    data_dir = Path(__file__).resolve().parents[2] / 'src/experiment/data'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    logger_ins = ExperimentLogger(save_path = log_dir)
    load_ins = load_datasets(base_dir = data_dir)
    split_ins = Manage3Split(seed = SEED)
    EMG_ins = EMG_preprocessing(fs = EMG_FREQ, bandpass_lowcut = EMG_LOWCUT, bandpass_highcut = EMG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    # EEG_ins = EEG_preprocessing(fs = EEG_FREQ, bandpass_lowcut = EEG_LOWCUT, bandpass_highcut = EEG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)

    #===========#
    # Load data #
    #===========#
    X_epoch_index, _, _ = load_ins.load_EMG_data(subject_name = subject_name, finger_name = 'index', EMG_config_dict = EMG_CONFIG_DICT, reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EMG_ins.preprocessing_routine)
    X_epoch_thumb, _, _ = load_ins.load_EMG_data(subject_name = subject_name, finger_name = 'thumb', EMG_config_dict = EMG_CONFIG_DICT, reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EMG_ins.preprocessing_routine)

    FREQ = RMS_FREQ

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
    print('Test 1 segment : ', X_test[0, 0:10, 2])
    print('Test 2 segment : ', X_test[7, 30:40, 1])
    print('Test 3 segment : ', X_test[17, 90:100, 0])'''