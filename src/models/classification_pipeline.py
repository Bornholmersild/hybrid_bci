# Classification
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset
import sherpa

# Manage datasets
import numpy as np
import pandas as pd

# Manage utils
import os 
from pathlib import Path
from datetime import datetime

# Own implementations
from src.utilities.preprocessing import EEG_preprocessing, EMG_preprocessing, RejectBadEpochs, Filtering #E402
from src.utilities.trainer_and_evaluator import FusionNet_train_eval, SingleNet_train_eval
from src.utilities.load_and_visualize_data import load_datasets

#==================#
# Global variables #
#==================#
EMG_FREQ = 2000
EEG_FREQ = 125

EMG_LOWCUT = 20
EMG_HIGHCUT = 450
EEG_LOWCUT = 0.05
EEG_HIGHCUT = 32

EEG_NUM_CH = 3
EEG_NUM_CH = 16

TRIAL_PERIOD = 9
TRIM_PERIOD = 3

EEG_USEABLE_CHANNELS = [2, 3, 6, 7, 10, 11]

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

EMG_CONFIG_DICT = {
    'rms_windowsize' : 32,
    'rms_stepsize' : 16,
    'hampel_windowsize' : 100,
    'hampel_sigma' : 2.0,
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
        self.rng = np.random.default_rng(SEED)

        print('data shape:', self.data.shape)
        print('labels shape:', self.labels.shape)
    
    def create_test_train_dataset(self, train_procent = 0.8):
        '''
        Return:
            train_dataset - 
            test_dataset -
        '''
        N_labels = len(self.labels)
        indices = self.rng.permutation(N_labels)            # Shuffle labels
        
        train_size = int(train_procent * N_labels)
        train_idx = indices[:train_size]
        test_idx  = indices[train_size:]

        return train_idx, test_idx
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_rng_generator(self):
        return self.rng

class MultiManageDataset(torch.utils.data.Dataset):
    def __init__(self, eeg, emg, labels):
        self.eeg = torch.tensor(eeg, dtype=torch.float32)
        self.emg = torch.tensor(emg, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.rng = np.random.default_rng(SEED)

        print('eeg shape:', self.eeg.shape)
        print('emg shape:', self.emg.shape)
        print('labels shape:', self.labels.shape)
    
    def create_test_train_dataset(self, train_procent = 0.8):
        N_labels = len(self.labels)
        indices = self.rng.permutation(N_labels)            # Shuffle labels
        
        train_size = int(train_procent * N_labels)
        train_idx = indices[:train_size]
        test_idx  = indices[train_size:]

        return train_idx, test_idx
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg[idx], self.emg[idx], self.labels[idx]

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

def load_EMG_data(subject_name : str | list, finger_name : str):
    base_dir = Path().resolve() / 'src/experiment/data'

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

def prepare_classes_single_sensor(epoch_data_class1, epoch_data_class2, fs = 125):

    # Segment for class 1
    rest_period_class1 = epoch_data_class1[:, :3*fs, :].copy()
    contract_period_class1 = epoch_data_class1[:, 3*fs : 6*fs, :].copy()
    release_period_class1 = epoch_data_class1[:, 6*fs:, :].copy()

    # Segment for class 2
    rest_period_class2 = epoch_data_class2[:, :3*fs, :].copy()
    contract_period_class2 = epoch_data_class2[:, 3*fs : 6*fs, :].copy()
    release_period_class2 = epoch_data_class2[:, 6*fs:, :].copy()

    rest_concat = np.concatenate( (rest_period_class1,
                                   rest_period_class2))

    X = np.concatenate( (contract_period_class1,
                         release_period_class1,
                         contract_period_class2,
                         release_period_class2,
                         rest_concat))

    labels = np.concatenate( (np.zeros(contract_period_class1.shape[0]), 
                            np.ones(release_period_class1.shape[0]),
                            np.ones(contract_period_class2.shape[0]) + 1,
                            np.ones(release_period_class2.shape[0]) + 2,
                            np.ones(rest_concat.shape[0]) + 3)
                            )
    
    return X, labels
    

def load_classfication(subject_name : str | list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    pin_memory = torch.cuda.is_available()              # Use pin_memory if CUDA is available
    # print("Pin memory set to:", pin_memory)

    LOG_NAME = f'{subject_name}_SingleNet_EMG_5classes'
    log_dir = os.path.abspath(os.path.join(os.getcwd(), f'loggings\{LOG_NAME}'))    # Path to log results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    #===========#
    # Load data #
    #===========#
    RMS_index, _ = load_EMG_data(subject_name = subject_name, finger_name = 'index')
    RMS_thumb, _ = load_EMG_data(subject_name = subject_name, finger_name = 'thumb')

    X, labels = prepare_classes_single_sensor(epoch_data_class1 = RMS_index, epoch_data_class2 = RMS_thumb)


    #====================================#
    # SHERPA Hyperparameter Optimazation #
    #====================================#
    parameters = [sherpa.Continuous(name='learning_rate', range=[0.0001, 0.001], scale='log'),
              sherpa.Continuous(name='dropout', range=[0., 0.4]),
              sherpa.Ordinal(name='batch_size', range=[16, 32, 64]),
              sherpa.Discrete(name='num_hidden_units', range=[32, 64]),         # before 64
              sherpa.Choice(name='activation', range=['relu', 'elu']),
              sherpa.Ordinal(name='lstm_layers', range=[1, 3])]

    # Hyperparameter optimization algorithms
    max_num_trials = 100
    algorithm = sherpa.algorithms.RandomSearch(max_num_trials = max_num_trials)

    # Study represents the hyperparameter optimization itself
    study = sherpa.Study(
        parameters = parameters,
        algorithm = algorithm,
        lower_is_better = True,
        disable_dashboard = True
    )

    #=======================#
    # Multi fusion datasets #
    #=======================#
    # dataset_ins = MultiManageDataset(EEG_X, EMG_X, labels)
    # train_eval_ins = FusionNet_train_eval()

    #=================#
    # Single datasets #
    #=================#
    dataset_ins = SingleManageDataset(data = X, labels = labels)
    train_eval_ins = SingleNet_train_eval()

    #========================================================#
    # THESE PARAMETERS ARE CHANCEABLE, DEPENDING ON THE TASK #
    #========================================================#
    DATA_CH = X.shape[2]
    NUM_CLASSES = 5

    train_idx, test_idx = dataset_ins.create_test_train_dataset(train_procent = 0.8)

    num_epochs = 200
    patience = 40                   # Early stopping patience - 25

    for trial in study:
        lr = trial.parameters['learning_rate']
        dropout = trial.parameters['dropout']       
        batch_size = trial.parameters['batch_size']
        num_hidden_units = trial.parameters['num_hidden_units']
        activation = trial.parameters['activation'] 
        lstm_layers = trial.parameters['lstm_layers']

        #=======================#
        # Multi fusion datasets #
        #=======================#
        # model = FusionNet(eeg_ch = 6, emg_ch = 3, hidden = num_hidden_units, lstm_layers = 1, num_classes = NUM_CLASSES, dropout = dropout, activation = activation)
        
        #=================#
        # Single datasets #
        #=================#
        model = SingleNet(data_ch = DATA_CH, hidden = num_hidden_units, lstm_layers = lstm_layers, num_classes = NUM_CLASSES, dropout = dropout, activation = activation)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)

        # Make a subset of data with desired indicies
        train_dataset = Subset(dataset_ins, train_idx)
        test_dataset  = Subset(dataset_ins, test_idx)

        # DataLoaders (update batch_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=0)
        
        best_vloss = float('inf')
        best_vacc = 0.0
        early_stopping_counter = 0

        # Create log folder
        log_folder = os.path.join(log_dir, f"trial_{trial.id}")
        os.makedirs(log_folder, exist_ok=False)
        writer = SummaryWriter(os.path.join(log_folder, 'trial_{}_timestamp_{}'.format(trial.id, timestamp)))

        for epoch in range(num_epochs):
            #print('EPOCH {}/{} at trial {}/{}:'.format(epoch + 1, num_epochs, trial.id, max_num_trials))

            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)

            # Train model
            avg_loss = train_eval_ins.train_one_epoch(model = model, train_loader = train_loader, criterion = criterion, optimizer = optimizer, device = device)

            # Set the model to evaluation mode
            model.eval()

            avg_vloss, vacc, _ = train_eval_ins.evaluate_one_epoch(model = model, test_loader = test_loader, criterion = criterion, device = device)

            #print('LOSS train {} valid {} and accuracy {}'.format(avg_loss, avg_vloss, vacc))

            # Tensor Board logging
            writer.add_scalars('Loss', { 'Training' : avg_loss, 'Validation' : avg_vloss }, epoch + 1)
            writer.add_scalars('Accuracy Validation', {'Validation' : vacc }, epoch + 1)
            writer.flush()

            study.add_observation(trial = trial,
                                iteration = epoch,
                                objective = avg_vloss,
                                context={'avg_train_loss': avg_loss})

            
            # Track best performance, and save the model's state
            if best_vacc < vacc:
                best_vacc = vacc

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                early_stopping_counter = 0
                # print("New best model found at epoch {} with validation loss: {:.4f}".format(epoch + 1, avg_vloss))
                
                if vacc > 60:
                    torch.save({
                        "model_state": model.state_dict(),
                        "accuracy": best_vacc,
                        "model_args": {
                            "data_ch": DATA_CH,
                            "hidden": num_hidden_units,
                            "lstm_layers": lstm_layers,
                            "num_classes": NUM_CLASSES,
                            "dropout": dropout,
                            "activation": activation,}}, r'{}\model.pth'.format(log_folder))
                    # print('model saved')

            else:
                early_stopping_counter += 1
                # print("Early stopping counter: {}/{}".format(early_stopping_counter, patience))

                if early_stopping_counter >= patience:
                    # print("Early stopping triggered after {} epochs without improvement.".format(early_stopping_counter))
                    break
            print(
                f'{subject_name} | '
                f'Trial {trial.id}/{max_num_trials} | '
                f'Epoch {epoch+1}/{num_epochs} | '
                f'Train {avg_loss:.4f} | '
                f'Val {avg_vloss:.4f} | '
                f'Acc {vacc:.2f} |',
                f'Early stopping {early_stopping_counter}',
                end='\r',
                flush=True
            )

        writer.close()
        study.save(log_dir)
        study.finalize(trial, status = 'COMPLETED')

    #----------------------------------------------------#
    # Document results from study trials and best version
    #----------------------------------------------------#
    df = pd.read_csv(f"{log_dir}\\results.csv")
    best = study.get_best_result()

    summary_row = {
        "Trial-ID" : best['Trial-ID'],
        "Status" : "BEST",
        'Iteration' : best['Iteration'],
        'activation' : best['activation'],
        'batch_size' : best['batch_size'],
        'dropout' : best['dropout'],
        'learning_rate' : best['learning_rate'],
        'num_hidden_units' : best['num_hidden_units'],
        'lstm_layers' : best['lstm_layers'],
        "Objective" : best['Objective'],
        "avg_train_loss" : best['avg_train_loss'],
    }

    df_new = pd.concat([pd.DataFrame([summary_row]), df], ignore_index=True)
    df_new.to_csv(f"{log_dir}\\results.csv", index=False)

def main():
    import time
    t0 = time.time()
    subjects = ['subject_0']

    for subject in subjects:
        load_classfication(subject_name = subject)

    print('Classification COMPLETE\n'
          'Time it took: ', time.time() - t0, 's')
    
if __name__ == '__main__':
    main()