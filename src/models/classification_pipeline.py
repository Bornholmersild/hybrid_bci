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
EEG_LOWCUT = 8
EEG_HIGHCUT = 30

EEG_NUM_CH = 3
EEG_NUM_CH = 16

TRIAL_PERIOD = 9
TRIM_PERIOD = 3

RMS_SAMPLING_WINDOW = 500           # 500 samples - 250 ms                      32 samples - 16 ms                                       
RMS_WINDOW_STEPSIZE = 50            # 50 samples - 25 ms (90 % overlap)         16 samples - 8 ms (50 % overlap)

HAMPEL_WINDOWSIZE = 100
HAMPEL_SIGMA = 2

EEG_USEABLE_CHANNELS = [2, 3, 6]

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

#===============================================#
# Class for traning either on EMG or EEG data #
#===============================================#
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
    bidirectional : bool (Optional hyperparameter)
        Enable bi-directional LSTM. Doubles the hidden_dim dimensionality.
    dropout : float (Optional hyperparameter)
        Introduce a dropout probability on the outputs of each LSTM layers. Except for the final layer.
    dense_ratio : float (Optional hyperparameter)
        Embed higher-dimensional space (dense_hidden_layers < dense_layers) -> learns more complex nonlienar combinations -> risk of overfitting.
        Distill LSTM features into a compact representation (dense_hidden_layers > dense_layers) -> strong regularization -> risk of underfit and removal of important patterns
    '''
    def __init__(self, input_dim : np.ndarray, output_dim : int, hidden_dim : int, lstm_layers : int, bidirectional : bool, dropout : float, activation : str, dense_ratio : float):
        super().__init__()
        self.bidirectional = bidirectional

        dense_hidden_layers = hidden_dim * 2 if bidirectional else hidden_dim
        dense_layers = max(8, int(dense_hidden_layers * dense_ratio))

        self.lstm = nn.LSTM(input_size = input_dim, 
                            hidden_size = hidden_dim,            # Don't consider bidirectional for hidden units. 
                            num_layers = lstm_layers, 
                            batch_first = True,                  # Data input will be (batch, seq_len, channels)
                            dropout = dropout if lstm_layers > 1 else 0,
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

class SingleNet_CNN_LSTM(nn.Module):
    '''
    Single network to perform CNN + LSTM on EEG or EMG datasets.\n
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
    bidirectional : bool (Optional hyperparameter)
        Enable bi-directional LSTM. Doubles the hidden_dim dimensionality.
    dropout : float (Optional hyperparameter)
        Introduce a dropout probability on the outputs of each LSTM layers. Except for the final layer.
    dense_ratio : float (Optional hyperparameter)
        Embed higher-dimensional space (dense_hidden_layers < dense_layers) -> learns more complex nonlienar combinations -> risk of overfitting.
        Distill LSTM features into a compact representation (dense_hidden_layers > dense_layers) -> strong regularization -> risk of underfit and removal of important patterns
    '''
    def __init__(self, input_dim : np.ndarray, output_dim : int, hidden_dim : int, lstm_layers : int, bidirectional : bool, dropout : float, activation : str, dense_ratio : float, cnn_filters = 32, kernel_size = 5):
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

        dense_hidden_layers = hidden_dim * 2 if bidirectional else hidden_dim  
        dense_layers = max(8, int(dense_hidden_layers * dense_ratio))

        activations = {
            'relu' : nn.ReLU,
            'elu' : nn.ELU
        }
        act = activations[activation]

        # ---- CNN MODULE ----
        # Figure out:
        #   AvgPool vs maxpool
        #   cnn_filters*2
        # Check it works with dimensions
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels = input_dim, out_channels = cnn_filters, kernel_size = kernel_size, padding = kernel_size // 2),        
            nn.BatchNorm1d(num_features = cnn_filters),
            act(),
            nn.MaxPool1d(kernel_size = 2),

            nn.Conv1d(in_channels = cnn_filters, out_channels = cnn_filters*2, kernel_size = kernel_size, padding = kernel_size // 2),        
            nn.BatchNorm1d(num_features = cnn_filters*2),
            act(),           
            nn.MaxPool1d(kernel_size = 2),
            
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size = cnn_filters*2,
            hidden_size = hidden_dim,
            num_layers = lstm_layers,
            batch_first = True,
            dropout = dropout if lstm_layers > 1 else 0,
            bidirectional = bidirectional
        )

        ''' FOR adding attension
        # __init__
        self.attension = Attension(hidden_dim = dense_hidden_layers)
        # forword
        lstm_out, _ = self.lstm(x)       # (batch, seq_len, hidden)
        context, attn_weights = self.attension(lstm_out)
        logits = self.classifier(context)'''
        
        self.classifier = nn.Sequential(
            nn.Linear(dense_hidden_layers, dense_layers),            # 64
            act(),
            nn.Dropout(dropout),
            nn.Linear(dense_layers, output_dim)
        )
    
    def forward(self, data : torch.Tensor):
        '''
        Runs the data through CNN + LSTM + Fully Connected layer

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

        x = data.permute(0, 2, 1)        # (B, S, CH) -> (B, CH, S)
        x = self.cnn(x)                  # (B, C_filter, S/2)
        x = x.permute(0, 2, 1)           # (B, C_filter, S/2) -> (B, S/2, C_filter)

        _, (hn, _) = self.lstm(x)        # (B, S/2, H)

        if self.bidirectional:                # Last two elements contain final forward and final reverse hidden states
            h_forward = hn[-2, :, :]          # (1, batch, hidden)
            h_backward = hn[-1, :, :]         # (1, batch, hidden)
            h_final = torch.cat((h_forward, h_backward), dim = 1)       # Concat -> (batch, hidden * 2)
        else:
            h_final = hn[-1]                  # (batch, hidden)

        logits = self.classifier(h_final)


        return logits, None, None        # None is placeholders

#======================================#
# Class for traning on EMG or EEG data #
#===============?======================#
class FusionNet_LSTM(nn.Module):
    def __init__(self, 
                 eeg_dim : int,
                 emg_dim : int,
                 eeg_output_dim : int, 
                 emg_output_dim : int, 
                 output_dim : int, 
                 hidden_dim : int, 
                 lstm_layers : int, 
                 bidirectional : bool, 
                 dropout : float, 
                 activation : str, 
                 dense_ratio : float):
        super().__init__()

        self.bidirectional = bidirectional

        #=========================================================================#
        # NOTE: Only used when EMG has no bidirectional and 1 lstm layer          #
        # Else:                                                                   #
        #   dense_hidden_layers = hidden_dim * 2 if bidirectional else hidden_dim #
        #   dense_layers = max(8, int(dense_hidden_layers * dense_ratio))         #
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        #   nn.Linear(dense_hidden_layers, dense_layers)                          #            
        #   nn.Linear(dense_layers, X_output_dim)                                 #
        #=========================================================================#
        EEG_dense_hidden_layers = hidden_dim * 2 if bidirectional else hidden_dim
        EEG_dense_layers = max(8, int(EEG_dense_hidden_layers * dense_ratio))           # Consider bidirection
        EMG_dense_layers = max(8, int(hidden_dim * dense_ratio))                        # Don't consider bidirection

        self.eeg_lstm = nn.LSTM(input_size = eeg_dim, 
                            hidden_size = hidden_dim,            # Don't consider bidirectional for hidden units. 
                            num_layers = lstm_layers, 
                            batch_first = True,                  # Data input will be (batch, seq_len, channels)
                            dropout = dropout if lstm_layers > 1 else 0,
                            bidirectional = bidirectional)       # Input dim (batch size, sequence length, input_size)
        
        self.emg_lstm = nn.LSTM(input_size = emg_dim, 
                            hidden_size = hidden_dim,            # Don't consider bidirectional for hidden units. 
                            num_layers = 1, 
                            batch_first = True,                  # Data input will be (batch, seq_len, channels)
                            dropout = dropout if lstm_layers > 1 else 0,
                            bidirectional = False)       # Input dim (batch size, sequence length, input_size)

        # WHAT OTHER ACTIVATIONS IS OF INTERST?
        activations = {
            'relu' : nn.ReLU,
            'elu' : nn.ELU
        }
        act = activations[activation]

        self.eeg_dense = nn.Sequential(
            nn.Linear(EEG_dense_hidden_layers, EEG_dense_layers),            # 64
            act(),
            nn.Dropout(dropout),
            nn.Linear(EEG_dense_layers, eeg_output_dim)
        )

        self.emg_dense = nn.Sequential(
            nn.Linear(hidden_dim, EMG_dense_layers),            # 64
            act(),
            nn.Dropout(dropout),
            nn.Linear(EMG_dense_layers, emg_output_dim)
        )

        self.fusion_dense = nn.Sequential(
            nn.Linear(emg_output_dim + eeg_output_dim, (emg_output_dim + eeg_output_dim)*2),            # 8 -> 16
            act(),
            nn.Dropout(dropout),
            nn.Linear((emg_output_dim + eeg_output_dim)*2, output_dim)                     # 16 -> 5
        )

    def _extract_hidden(self, hn):
        """Extract final hidden state from LSTM"""
        if self.bidirectional:
            h_forward = hn[-2]                                          # (1, batch, hidden)
            h_backward = hn[-1]                                         # (1, batch, hidden)
            h_final = torch.cat((h_forward, h_backward), dim=1)         # Concat -> (batch, hidden * 2)
        else:
            h_final = hn[-1]                                            # (batch, hidden)

        return h_final

    def forward(self, eeg, emg):

        # EEG branch
        _, (hn_eeg, _) = self.eeg_lstm(eeg)             # Final LSTM hidden layer : hn (bi * num_layers, batch, hidden) - bi = 2 if bidirectional == True
        h_eeg = self._extract_hidden(hn_eeg)            # Last two elements contain final forward and final reverse hidden states
        eeg_logits = self.eeg_dense(h_eeg)

        # EMG branch
        _, (hn_emg, _) = self.emg_lstm(emg)
        h_emg = hn_emg[-1]                              # NOTE : With bidirectional -> h_emg = self._extract_hidden(hn_emg)
        emg_logits = self.emg_dense(h_emg)

        # Late fusion
        fusion_input = torch.cat([eeg_logits, emg_logits], dim=1)

        fusion_logits = self.fusion_dense(fusion_input)

        return fusion_logits, eeg_logits, emg_logits, None, None

class FusionNet_CNN_LSTM(nn.Module):
    '''
    Single network to perform CNN + LSTM on EEG or EMG datasets.\n
    Returns logits from LSTM features to number of classes to classify

    Parameters
    ----------
    eeg_dim : int
        The number of expected features in the input sequence at each time step (n_channels)
    emg_dim : int
        The number of expected features in the input sequence at each time step (n_channels)
    eeg_output_dim : int
        The expected amount of classes to be predicted
    emg_output_dim : int
        The expected amount of classes to be predicted
    output_dim : int
        Maps the hidden state in nn.Linear outputs to predictions (n_classes)
    hidden_dim : int (Optional hyperparameter)
        The number of features in the hidden state. How much memory should present the hidden state at one time stamp
    lstm_layers : int (Optional hyperparameter)
        Stacking multiple LSTM layers deepens the model. If is not in the timestamp direction. But the hiddenstate goes into the input of another LSTM.
    bidirectional : bool (Optional hyperparameter)
        Enable bi-directional LSTM. Doubles the hidden_dim dimensionality.
    dropout : float (Optional hyperparameter)
        Introduce a dropout probability on the outputs of each LSTM layers. Except for the final layer.
    activation : str (Optional hyperparameter)
        Activation function used in the model
    dense_ratio : float (Optional hyperparameter)
        Embed higher-dimensional space (dense_hidden_layers < dense_layers) -> learns more complex nonlienar combinations -> risk of overfitting.
        Distill LSTM features into a compact representation (dense_hidden_layers > dense_layers) -> strong regularization -> risk of underfit and removal of important patterns
    dense_fusion_layer : int
        Final dense layer given by eeg_output_dim + emg_output_dim -> dense_fusion_layer
    cnn_filters : int (Optional hyperparameter)
        Filters in the CNN 
    kernel_size : int
        Kernel size in the CNN
    '''
    def __init__(self,
                 eeg_dim : int,
                 emg_dim : int,
                 eeg_output_dim : int,
                 emg_output_dim : int,
                 output_dim : int,
                 hidden_dim : int,
                 lstm_layers : int,
                 bidirectional : bool,
                 dropout : float,
                 activation : str,
                 dense_ratio : float,
                 cnn_filters : int,
                 eeg_kernel_size : int,
                 emg_kernel_size : int):
        super().__init__()
        self.bidirectional = bidirectional

        #=========================================================================#
        # NOTE: Only used when EMG has no bidirectional and 1 lstm layer          #
        # Else:                                                                   #
        #   dense_hidden_layers = hidden_dim * 2 if bidirectional else hidden_dim #
        #   dense_layers = max(8, int(dense_hidden_layers * dense_ratio))         #
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        #   nn.Linear(dense_hidden_layers, dense_layers)                          #            
        #   nn.Linear(dense_layers, X_output_dim)                                 #
        #=========================================================================#
        EEG_dense_hidden_layers = hidden_dim * 2 if bidirectional else hidden_dim
        EEG_dense_layers = max(8, int(EEG_dense_hidden_layers * dense_ratio))           # Consider bidirection
        EMG_dense_layers = max(8, int(hidden_dim * dense_ratio))                        # Don't consider bidirection

        activations = {
            'relu' : nn.ReLU,
            'elu' : nn.ELU
        }
        act = activations[activation]

        self.eeg_cnn = nn.Sequential(
            nn.Conv1d(in_channels = eeg_dim, out_channels = cnn_filters, kernel_size = eeg_kernel_size, padding = eeg_kernel_size // 2),        
            nn.BatchNorm1d(num_features = cnn_filters),
            act(),
            nn.MaxPool1d(kernel_size = 2),

            nn.Conv1d(in_channels = cnn_filters, out_channels = cnn_filters*2, kernel_size = eeg_kernel_size, padding = eeg_kernel_size // 2),        
            nn.BatchNorm1d(num_features = cnn_filters*2),
            act(),           
            nn.MaxPool1d(kernel_size = 2),
            
            nn.Dropout(dropout)
        )

        self.emg_cnn = nn.Sequential(
            nn.Conv1d(in_channels = emg_dim, out_channels = cnn_filters, kernel_size = emg_kernel_size, padding = emg_kernel_size // 2),        
            nn.BatchNorm1d(num_features = cnn_filters),
            act(),
            nn.MaxPool1d(kernel_size = 2),

            nn.Conv1d(in_channels = cnn_filters, out_channels = cnn_filters*2, kernel_size = emg_kernel_size, padding = emg_kernel_size // 2),        
            nn.BatchNorm1d(num_features = cnn_filters*2),
            act(),           
            nn.MaxPool1d(kernel_size = 2),
            
            nn.Dropout(dropout)
        )

        self.eeg_lstm = nn.LSTM(input_size = cnn_filters*2, 
                            hidden_size = hidden_dim,            # Don't consider bidirectional for hidden units. 
                            num_layers = lstm_layers, 
                            batch_first = True,                  # Data input will be (batch, seq_len, channels)
                            dropout = dropout if lstm_layers > 1 else 0,
                            bidirectional = bidirectional)       # Input dim (batch size, sequence length, input_size)
        
        self.emg_lstm = nn.LSTM(input_size = cnn_filters*2, 
                            hidden_size = hidden_dim,            # Don't consider bidirectional for hidden units. 
                            num_layers = 1, 
                            batch_first = True,                  # Data input will be (batch, seq_len, channels)
                            dropout = dropout if lstm_layers > 1 else 0,
                            bidirectional = False)       # Input dim (batch size, sequence length, input_size)
        
        self.eeg_dense = nn.Sequential(
            nn.Linear(EEG_dense_hidden_layers, EEG_dense_layers),            # 64
            act(),
            nn.Dropout(dropout),
            nn.Linear(EEG_dense_layers, eeg_output_dim)
        )

        self.emg_dense = nn.Sequential(
            nn.Linear(hidden_dim, EMG_dense_layers),            # 64
            act(),
            nn.Dropout(dropout),
            nn.Linear(EMG_dense_layers, emg_output_dim)
        )

        self.fusion_dense = nn.Sequential(
            nn.Linear(emg_output_dim + eeg_output_dim, (emg_output_dim + eeg_output_dim) * 2),            # 8 -> 16
            act(),
            nn.Dropout(dropout),
            nn.Linear((emg_output_dim + eeg_output_dim)*2, output_dim)                     # 16 -> 5
        )
    
    def _extract_hidden(self, hn):
        """Extract final hidden state from LSTM"""
        if self.bidirectional:
            h_forward = hn[-2]                                          # (1, batch, hidden)
            h_backward = hn[-1]                                         # (1, batch, hidden)
            h_final = torch.cat((h_forward, h_backward), dim=1)         # Concat -> (batch, hidden * 2)
        else:
            h_final = hn[-1]                                            # (batch, hidden)

        return h_final
    
    def forward(self, eeg : torch.Tensor, emg : torch.Tensor):
        '''
        Runs the data through CNN + LSTM + Fully Connected layer

        Parameters
        ----------
        eeg : torch.Tensor
            DataLoader tensor with dimension: (batch, seq_len, channels)
        emg : torch.Tensor
            DataLoader tensor with dimension: (batch, seq_len, channels)

        Returns
        ----------
        fusion_logits : torch.Tensor
            Linear transform of LSTM features for eeg and emg -> logits
        eeg_logits : torch.Tensor
            Linear transform of LSTM features for eeg -> logits
        emg_logits : torch.Tensor
            Linear transform of LSTM features for emg -> logits
        _ : None
            Placeholder for Attension
        _ : None
            Placeholder for Attension
        '''
        # EEG branch
        x_eeg = eeg.permute(0, 2, 1)           # (B, S, CH) -> (B, CH, S)
        x_eeg = self.eeg_cnn(x_eeg)            # (B, C_filter, S/4)
        x_eeg = x_eeg.permute(0, 2, 1)         # (B, C_filter, S/4) -> (B, S/4, C_filter)
        _, (hn_eeg, _) = self.eeg_lstm(x_eeg)  # (B, S/4, H)
        h_eeg = self._extract_hidden(hn_eeg)   # Last two elements contain final forward and final reverse hidden states
        eeg_logits = self.eeg_dense(h_eeg)

        # EMG branch
        x_emg = emg.permute(0, 2, 1)           # (B, S, CH) -> (B, CH, S)
        x_emg = self.emg_cnn(x_emg)            # (B, C_filter, S/4)
        x_emg = x_emg.permute(0, 2, 1)         # (B, C_filter, S/4) -> (B, S/4, C_filter)
        _, (hn_emg, _) = self.emg_lstm(x_emg)  # (B, S/4, H)
        h_emg = hn_emg[-1]
        emg_logits = self.emg_dense(h_emg)

        # Late fusion
        fusion_input = torch.cat([eeg_logits, emg_logits], dim=1)

        fusion_logits = self.fusion_dense(fusion_input)

        return fusion_logits, eeg_logits, emg_logits, None, None       # None is placeholders

#=================#
# Handles dataset #
#=================#

class SingleManageDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, data_type):
        '''
        Takes in the concatinated dataset of all trials, samples and channels.
        Args:
            X [ndArray] - with the dimension of (trials, samples, channels)
            y [int] - Indicate the number of trials 
        '''
        if str.upper(data_type) == 'EEG':
            labels = self._map_to_emg_labels(labels = labels)
        # Convert to tensors
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

        print('data shape:', self.data.shape)
        print('labels shape:', self.labels.shape)
        print()
    
    def _map_to_emg_labels(self, labels):
        '''
        Only applied to EEG dataset
        '''
        map_labels = labels.copy()

        map_labels[(labels == 0) | (labels == 2)] = 0
        map_labels[(labels == 1) | (labels == 3)] = 1
        map_labels[labels == 4] = 2

        return map_labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class MultiManageDataset(torch.utils.data.Dataset):
    def __init__(self, eeg, emg, eeg_labels, emg_labels):
        map_labels = self._map_to_emg_labels(eeg_labels)

        self.eeg = torch.tensor(eeg, dtype=torch.float32)
        self.emg = torch.tensor(emg, dtype=torch.float32)
        self.eeg_labels = torch.tensor(map_labels, dtype=torch.long)
        self.emg_labels = torch.tensor(emg_labels, dtype=torch.long)

        print('eeg shape:', self.eeg.shape)
        print('emg shape:', self.emg.shape)
        print('eeg labels shape:', self.eeg_labels.shape)
        print('emg labels shape:', self.emg_labels.shape)
    
    def _map_to_emg_labels(self, labels):
    
        map_labels = labels.copy()

        map_labels[(labels == 0) | (labels == 2)] = 0
        map_labels[(labels == 1) | (labels == 3)] = 1
        map_labels[labels == 4] = 2

        return map_labels
    
    def __len__(self):
        return len(self.eeg_labels)

    def __getitem__(self, idx):
        return self.eeg[idx], self.emg[idx], self.eeg_labels[idx], self.emg_labels[idx]

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
    
    def build_modality_split(self, num_index_trials : int, num_thumb_trials : int, epoch_index : np.ndarray, epoch_thumb : np.ndarray, fs : int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Random shuffle indicies for trials and divide into train, validation and test dataset.

        Parameters
        ----------
        num_index_trials : int
            Number of trials from X_epoch_index.shape[0]
        num_thumb_trials : int
            Number of trials from X_epoch_thumb.shape[0]
        epoch_index : np.ndarray
            Array of data given in epochs. Shape (Epochs, sequence, channels)
        epoch_thumb : np.ndarray
            Array of data given in epochs. Shape (Epochs, sequence, channels)
        fs : int
            Sampling frequency for either EEG or EMG

        Returns
        ----------
        X_train : np.ndarray
        X_val : np.ndarray 
        X_test : np.ndarray
        y_train : np.ndarray
        y_val : np.ndarray
        y_test : np.ndarray 
        '''
        train_i, val_i, test_i = self._split_trials(num_index_trials)
        train_t, val_t, test_t = self._split_trials(num_thumb_trials)

        # EEG train, validation and test datasplit
        X_train, y_train = self._build_split(                   # Train data split
            epoch_index = epoch_index,
            epoch_thumb = epoch_thumb,
            index_trials_indices = train_i,
            thumb_trials_indices = train_t,
            fs = fs)
        
        X_val, y_val = self._build_split(                       # Validation data split
            epoch_index = epoch_index,
            epoch_thumb = epoch_thumb,
            index_trials_indices = val_i,
            thumb_trials_indices = val_t,
            fs = fs)
        
        X_test, y_test = self._build_split(
            epoch_index = epoch_index,
            epoch_thumb = epoch_thumb,
            index_trials_indices = test_i,
            thumb_trials_indices = test_t,
            fs = fs)
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _build_split(self, 
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

        data_type : str
            Either EEG or EMG. Purpose -> To define different amount of classes

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
    
        X = np.concatenate([
            contract_i,
            release_i,
            contract_t,
            release_t,
            np.concatenate([rest_i, rest_t])
        ])

        y = np.concatenate([
            np.zeros(len(contract_i)),                    # 0 index_contract
            np.ones(len(release_i)),                      # 1 index_release
            np.full(len(contract_t), 2),                  # 2 thumb_contract
            np.full(len(release_t), 3),                   # 3 thumb_release
            np.full(len(rest_i)+len(rest_t), 4)           # 4 rest
        ])
    
        return X, y

    def _split_trials(self, num_trials : int, train_ratio : int = 0.7) -> tuple[list, list, list]:
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

#========================#
# Dynamic model handling #
#========================#
class SingleNetHandler:
    def __init__(self, model_name : str, sensor_name : str):
        self.model_name = model_name
        self.sensor_name = sensor_name

    def get_hyperparameters(self):
        '''
        Return parameters for a specfic model
        '''

        common_params = [
            # General
            sherpa.Continuous('learning_rate', [1e-5, 1e-3], scale='log'),      # try out 1e-5, 1e-1
            sherpa.Continuous("weight_decay", [1e-6, 1e-2], scale="log"),       # try out 1e-5, 1e-1
            sherpa.Continuous('dropout', [0.1, 0.5]),
            sherpa.Ordinal('batch_size', [16, 32, 64]),
            sherpa.Ordinal('dense_ratio', [0.25, 0.5, 0.75, 1.0]),
            sherpa.Choice('activation', ['relu', 'elu']),
            # LSTM
            sherpa.Ordinal('num_hidden_units', [32, 64, 128, 256]),
            sherpa.Choice("bidirectional", [False, True]),
            sherpa.Choice('lstm_layers', [1, 2, 3]),
        ]

        if self.model_name == "FusionNet_LSTM":
            return common_params

        elif self.model_name == "FusionNet_CNN_LSTM":
            cnn_params = [
                sherpa.Ordinal('cnn_filters', [16, 32, 64]),
                sherpa.Ordinal('EEG_kernel_ratio', [0.01, 0.02, 0.03, 0.04, 0.1]),
                sherpa.Ordinal('EMG_kernel_ratio', [0.03, 0.06, 0.09, 0.12, 0.15]),
            ]
            return common_params + cnn_params
        else:
            raise ValueError(f'model_name does not correspond to a model class : {self.model_name}')
    
class FusionNetHandler:
    def __init__(self, model_name : str):
        self.model_name = model_name

    def get_hyperparameters(self):
        '''
        Return parameters for a specfic model
        '''

        common_params = [
            # General
            sherpa.Continuous('learning_rate', [1e-5, 1e-3], scale='log'),      # try out 1e-5, 1e-1
            sherpa.Continuous("weight_decay", [1e-6, 1e-2], scale="log"),       # try out 1e-5, 1e-1
            sherpa.Continuous('dropout', [0.1, 0.5]),
            sherpa.Ordinal('batch_size', [16, 32, 64]),
            sherpa.Ordinal('dense_ratio', [0.25, 0.5, 0.75, 1.0]),
            sherpa.Choice('activation', ['relu', 'elu']),
            # LSTM
            sherpa.Ordinal('num_hidden_units', [32, 64, 128, 256]),
            sherpa.Choice("bidirectional", [False, True]),
            sherpa.Choice('lstm_layers', [1, 2, 3]),
        ]

        if self.model_name == "FusionNet_LSTM":
            return common_params

        elif self.model_name == "FusionNet_CNN_LSTM":
            cnn_params = [
                sherpa.Ordinal('cnn_filters', [16, 32, 64]),
                sherpa.Ordinal('EEG_kernel_ratio', [0.01, 0.02, 0.03, 0.04, 0.1]),
                sherpa.Ordinal('EMG_kernel_ratio', [0.03, 0.06, 0.09, 0.12, 0.15]),
            ]
            return common_params + cnn_params
        else:
            raise ValueError(f'model_name does not correspond to a model class : {self.model_name}')
        
    def build_model_config(self, trial : sherpa.Study, EEG_CH : int, EMG_CH : int, EEG_CLASSES : int, EMG_CLASSES : int, TOTAL_CLASSES : int, EEG_samples : int, EMG_samples : int):
        '''
        Build a config dict for model input
        '''
        config = {
            "eeg_dim": EEG_CH,
            "emg_dim": EMG_CH,
            "eeg_output_dim": EEG_CLASSES,
            "emg_output_dim": EMG_CLASSES,
            "output_dim": TOTAL_CLASSES,
            "hidden_dim": trial.parameters['num_hidden_units'],
            "lstm_layers": trial.parameters['lstm_layers'],
            "bidirectional": trial.parameters['bidirectional'],
            "dropout": trial.parameters['dropout'],
            "activation": trial.parameters['activation'],
            "dense_ratio": trial.parameters['dense_ratio'],
        }

        if self.model_name == "FusionNet_CNN_LSTM":
            config.update({
                "cnn_filters": trial.parameters['cnn_filters'],
                "eeg_kernel_size": self._kernel_from_ratio(
                    EEG_samples, trial.parameters['EEG_kernel_ratio'], min_kernel=3),
                "emg_kernel_size": self._kernel_from_ratio(
                    EMG_samples, trial.parameters['EMG_kernel_ratio'], min_kernel=3),
            })

        return config
    
    def build_training_config(self, trial : sherpa.Study):
        return {
            "lr": trial.parameters["learning_rate"],
            "weight_decay": trial.parameters["weight_decay"],
            "batch_size": trial.parameters["batch_size"],
        }
    
    def get_model(self, config: dict):
        if self.model_name == "FusionNet_LSTM":
            return FusionNet_LSTM(**config)

        elif self.model_name == "FusionNet_CNN_LSTM":
            return FusionNet_CNN_LSTM(**config)
    
    def _kernel_from_ratio(self, seq_len, ratio, min_kernel = 3):
        kernel_size = max(min_kernel, int(round(seq_len * ratio)))

        # Make odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Cannot exceed sequence length
        if kernel_size > seq_len:
            kernel_size = seq_len if seq_len % 2 == 1 else seq_len - 1
        
        return kernel_size

class ExperimentLogger:

    def __init__(self, save_path : Path):
        self.save_path = save_path / 'SHERPA_results.pt'
        
        # Load existing file if present
        if os.path.exists(self.save_path):
            self.results = torch.load(self.save_path)
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

#==============================#
# Traning of model Per subject #
#==============================#
def singleNet_classfication(subject_name : str | list, sherpa_log_folder : str = 'SingleNet_LSTM_EMG', data_type : str = None):
    # When chancing between EEG and EMG
    # preprocessing instance
    # Load function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()              # Use pin_memory if CUDA is available
    print(f"Using device: {device}")
    print("Pin memory set to:", pin_memory)

    LOG_NAME = f'{subject_name}'
    log_dir = Path(__file__).resolve().parent / f'loggings/{sherpa_log_folder}/{LOG_NAME}'         # Path(__file__).resolve() -> Absolute path to this file
    data_dir = Path(__file__).resolve().parents[2] / 'src/experiment/data'
    data_type = str.upper(data_type)
    #==========================#
    # NOTE: Tensorboard config #
    #==========================#
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')                                        # Use when having tensorboard
    os.makedirs(log_dir, exist_ok=False)                                                          # use without tensorboard

    logger_ins = ExperimentLogger(save_path = log_dir)
    load_ins = load_datasets(base_dir = data_dir)
    split_ins = Manage3Split(seed = SEED)
    
    if data_type == 'EMG':
        EMG_ins = EMG_preprocessing(fs = EMG_FREQ, bandpass_lowcut = EMG_LOWCUT, bandpass_highcut = EMG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    elif data_type == 'EEG':
        EEG_ins = EEG_preprocessing(fs = EEG_FREQ, bandpass_lowcut = EEG_LOWCUT, bandpass_highcut = EEG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    else:
        raise ValueError('data_type must be either EMG or EEG')

    #===========#
    # Load data #
    #===========#
    if data_type == 'EMG':
        X_epoch_index, _, _ = load_ins.load_EMG_data(subject_name = subject_name, finger_name = 'index', EMG_config_dict = EMG_CONFIG_DICT, reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EMG_ins.preprocessing_routine)
        X_epoch_thumb, _, _ = load_ins.load_EMG_data(subject_name = subject_name, finger_name = 'thumb', EMG_config_dict = EMG_CONFIG_DICT, reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EMG_ins.preprocessing_routine)
    else:
        X_epoch_index, _ = load_ins.load_EEG_data(subject_name = subject_name, finger_name = 'index', reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EEG_ins.preprocessing_routine, EEG_useable_channels = EEG_USEABLE_CHANNELS)
        X_epoch_thumb, _ = load_ins.load_EEG_data(subject_name = subject_name, finger_name = 'thumb', reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EEG_ins.preprocessing_routine, EEG_useable_channels = EEG_USEABLE_CHANNELS)

    num_index_trials = X_epoch_index.shape[0]
    num_thumb_trials = X_epoch_thumb.shape[0]

    FREQ = RMS_FREQ if data_type == 'EMG' else EEG_FREQ

    X_train, X_val, X_test, y_train, y_val, y_test = split_ins.build_modality_split(
        num_index_trials = num_index_trials,
        num_thumb_trials = num_thumb_trials,
        epoch_index = X_epoch_index,
        epoch_thumb = X_epoch_thumb,
        fs = FREQ
    )
    
    _, num_samples, num_channels = X_train.shape

    #=================#
    # Single datasets #
    #=================#
    train_eval_ins = SingleNet_train_eval()

    print('\nTraining dataset shapes:')
    train_dataset_ins = SingleManageDataset(X_train, y_train, data_type = data_type)
    print('Validation dataset shapes:')
    val_dataset_ins = SingleManageDataset(X_val, y_val, data_type = data_type)
    print('Testing dataset shapes:')
    test_dataset_ins = SingleManageDataset(X_test, y_test, data_type = data_type)

    #========================================================#
    # THESE PARAMETERS ARE CHANCEABLE, DEPENDING ON THE TASK #
    #========================================================#
    MAX_NUM_TRIALS = 150             # 75 - 250 (simply to max) 
    DATA_CH = num_channels
    NUM_CLASSES = 5 if data_type == 'EMG' else 3
    NUM_EPOCHS = 250                 # 150 - 200
    PATIENCE = 25 if data_type == 'EMG' else 50                   # Early stopping patience - 25
    NUM_INITIAL_DATA_POINTS = 20
    
    #===========#
    # Constants #
    #===========#
    global_best_vloss = float("inf")                # Used to only save one model.

    #====================================#
    # SHERPA Hyperparameter Optimazation #
    #====================================#

    parameters = [
        # General
        sherpa.Continuous(name='learning_rate', range=[0.00001, 0.001], scale='log'),
        sherpa.Continuous(name="weight_decay", range=[1e-6, 1e-2], scale="log"),  
        sherpa.Continuous(name='dropout', range=[0.1, 0.5]),
        sherpa.Ordinal(name='batch_size', range=[16, 32, 64]),
        sherpa.Ordinal(name='dense_ratio', range=[0.25, 0.5, 0.75, 1.0]),
        sherpa.Choice(name='activation', range=['relu', 'elu']),

        # LSTM
        sherpa.Ordinal(name='num_hidden_units', range=[32, 64, 128, 256]),          # 32, 64, 128, 256          
        sherpa.Choice(name="bidirectional", range=[False, True]),                      
        sherpa.Choice(name='lstm_layers', range=[1, 2, 3]),

        # CNN
        # sherpa.Ordinal(name='cnn_filters', range=[16, 32, 64]),
        # sherpa.Ordinal(name='EEG_kernel_ratio', range=[0.01, 0.02, 0.03, 0.04, 0.1]),   # EEG [3.75, 7.5, 11.25, 15, 37.5] samples
        # sherpa.Ordinal(name='EMG_kernel_ratio', range=[X]),   # EEG : 3.75, 7.5, 11.25, 15, 37.5
    ]
    
    # algorithm = sherpa.algorithms.RandomSearch(max_num_trials = MAX_NUM_TRIALS)
    algorithm = sherpa.algorithms.GPyOpt(
        max_num_trials = MAX_NUM_TRIALS,
        acquisition_type = 'EI',                     # Expected improvement
        num_initial_data_points = NUM_INITIAL_DATA_POINTS                 # Number of hyperparameter configurations before model learns
    )
    # Study represents the hyperparameter optimization itself
    study = sherpa.Study(
        parameters = parameters,
        algorithm = algorithm,
        lower_is_better = True,
        disable_dashboard = True
    )

    for trial in study:
        # General
        dropout = trial.parameters['dropout']       
        batch_size = trial.parameters['batch_size']
        activation = trial.parameters['activation'] 
        dense_ratio = trial.parameters['dense_ratio']
        weight_decay = trial.parameters["weight_decay"]
        lr = trial.parameters["learning_rate"]

        # LSTM
        num_hidden_units = trial.parameters['num_hidden_units']
        lstm_layers = trial.parameters['lstm_layers']
        bidirectional = trial.parameters['bidirectional']   
        
        # CNN
        # cnn_filters = trial.parameters['cnn_filters']
        # kernel_ratio = trial.parameters['kernel_ratio']
        # EEG_kernel_size = kernel_from_ratio(seq_len = EEG_num_samples, ratio = kernel_ratio, min_kernel = 3)
        # EMG_kernel_size = kernel_from_ratio(seq_len = EMG_num_samples, ratio = kernel_ratio, min_kernel = 3)
        # kernel_size = kernel_from_ratio(seq_len = num_samples, ratio = kernel_ratio, min_kernel = 3)

        #=================#
        # Single datasets #
        #=================#
        model = SingleNet_LSTM(input_dim = DATA_CH, output_dim = NUM_CLASSES, hidden_dim = num_hidden_units, lstm_layers = lstm_layers, bidirectional = bidirectional, dropout = dropout, activation = activation, dense_ratio = dense_ratio)
        # model = SingleNet_CNN_LSTM(input_dim = DATA_CH, output_dim = NUM_CLASSES, hidden_dim = num_hidden_units, lstm_layers = lstm_layers, bidirectional = bidirectional, dropout = dropout, activation = activation, dense_ratio = dense_ratio, cnn_filters = cnn_filters, kernel_size = kernel_size)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params = model.parameters(), lr = lr, weight_decay = weight_decay)

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

        #===================================#
        # NOTE: Tensorboard config          #
        #   Enable all if using tensorboard #
        #===================================#
        # log_folder = os.path.join(log_dir, f"trial_{trial.id}")               
        # os.makedirs(log_folder, exist_ok=False)
        # writer = SummaryWriter(os.path.join(log_folder, 'trial_{}_timestamp_{}'.format(trial.id, timestamp)))

        for epoch in range(NUM_EPOCHS):

            # Train model
            avg_train_loss = train_eval_ins.train_one_epoch(model = model, train_loader = train_loader, criterion = criterion, optimizer = optimizer, device = device)

            # Validate model
            avg_vloss, vacc, _ = train_eval_ins.validaton_one_epoch(model = model, val_loader = val_loader, criterion = criterion, device = device)

            # Tensor Board logging
            # writer.add_scalars('Loss', { 'Training' : avg_train_loss, 'Validation' : avg_vloss }, epoch + 1)
            # writer.add_scalars('Accuracy Validation', {'Validation' : vacc }, epoch + 1)
            # writer.flush()

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

        if best_val_loss < global_best_vloss :
            global_best_vloss = best_val_loss

            model_arg = {
            "input_dim": DATA_CH,
            "output_dim" : NUM_CLASSES,
            "hidden_dim": num_hidden_units,
            "lstm_layers": lstm_layers,
            "bidirectional" : bidirectional,
            "dropout": dropout,
            "activation": activation,
            "dense_ratio": dense_ratio,}
            ''' model_arg With CNN
            "cnn_filters" : cnn_filters,
            "kernel_size" : kernel_size,
        '''
            
            torch.save({
                "model_state": best_state_dict,
                "model_args": model_arg, 
                "optimizer_state_dict": best_optimizer_dict,
                "hyperparameters": trial.parameters,}, 
                r'{}\model.pth'.format(log_dir))
            
        # writer.close()                            # NOTE: Enable with tensorboard
        study.finalize(trial, status = 'COMPLETED')

def fusionNet_classfication(subject_name : str | list, sherpa_log_folder : str = 'fusionNet_LSTM', model_name : str = 'FusionNet_LSTM'):
    # When chancing between EEG and EMG
    # preprocessing instance
    # Load function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()              # Use pin_memory if CUDA is available
    print(f"Using device: {device}")
    print("Pin memory set to:", pin_memory)

    LOG_NAME = f'{subject_name}'
    log_dir = Path(__file__).resolve().parent / f'loggings/{sherpa_log_folder}/{LOG_NAME}'         # Path(__file__).resolve() -> Absolute path to this file
    data_dir = Path(__file__).resolve().parents[2] / 'src/experiment/data'
    
    #==========================#
    # NOTE: Tensorboard config #
    #==========================#
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')                                        # Use when having tensorboard
    os.makedirs(log_dir, exist_ok=False)    

    logger_ins = ExperimentLogger(save_path = log_dir)
    load_ins = load_datasets(base_dir = data_dir)
    split_ins = Manage3Split(seed = SEED)
    EMG_ins = EMG_preprocessing(fs = EMG_FREQ, bandpass_lowcut = EMG_LOWCUT, bandpass_highcut = EMG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    EEG_ins = EEG_preprocessing(fs = EEG_FREQ, bandpass_lowcut = EEG_LOWCUT, bandpass_highcut = EEG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    train_eval_ins = FusionNet_train_eval()
    model_handler_ins = FusionNetHandler(model_name = model_name)
    #===========#
    # Load data #
    #===========#
    EEG_epoch_index, EMG_epoch_index, _, _ = load_ins.load_EEG_EMG_data(subject_name = subject_name, finger_name = 'index', reject_config_dict = REJECT_CONFIG_DICT, EEG_preprocessing_func = EEG_ins.preprocessing_routine, EMG_preprocessing_func = EMG_ins.preprocessing_routine, EMG_config_dict = EMG_CONFIG_DICT, EEG_useable_channels = EEG_USEABLE_CHANNELS)
    EEG_epoch_thumb, EMG_epoch_thumb, _, _ = load_ins.load_EEG_EMG_data(subject_name = subject_name, finger_name = 'thumb', reject_config_dict = REJECT_CONFIG_DICT, EEG_preprocessing_func = EEG_ins.preprocessing_routine, EMG_preprocessing_func = EMG_ins.preprocessing_routine, EMG_config_dict = EMG_CONFIG_DICT, EEG_useable_channels = EEG_USEABLE_CHANNELS)

    num_index_trials = EEG_epoch_index.shape[0]
    num_thumb_trials = EEG_epoch_thumb.shape[0]

    X_EEG_train, X_EEG_val, X_EEG_test, y_EEG_train, y_EEG_val, y_EEG_test = split_ins.build_modality_split(
        num_index_trials = num_index_trials,
        num_thumb_trials = num_thumb_trials,
        epoch_index = EEG_epoch_index,
        epoch_thumb = EEG_epoch_thumb,
        fs = EEG_FREQ
    )

    X_EMG_train, X_EMG_val, X_EMG_test, y_EMG_train, y_EMG_val, y_EMG_test = split_ins.build_modality_split(
        num_index_trials = num_index_trials,
        num_thumb_trials = num_thumb_trials,
        epoch_index = EMG_epoch_index,
        epoch_thumb = EMG_epoch_thumb,
        fs = RMS_FREQ
    )

    _, EEG_num_samples, EEG_num_channels = X_EEG_train.shape
    _, EMG_num_samples, EMG_num_channels = X_EMG_train.shape

    #=======================#
    # Multi fusion datasets #
    #=======================#
    print('\nTraining dataset shapes:')
    train_dataset_ins = MultiManageDataset(X_EEG_train, X_EMG_train, y_EEG_train, y_EMG_train)
    print('Validation dataset shapes:')
    val_dataset_ins = MultiManageDataset(X_EEG_val, X_EMG_val, y_EEG_val, y_EMG_val)
    print('Testing dataset shapes:')
    test_dataset_ins = MultiManageDataset(X_EEG_test, X_EMG_test, y_EEG_test, y_EMG_test)

    #========================================================#
    # THESE PARAMETERS ARE CHANCEABLE, DEPENDING ON THE TASK #
    #========================================================#
    MAX_NUM_TRIALS = 100             # 75 - 250 (simply to max) 
    NUM_INITIAL_DATA_POINTS = 20
    EEG_CH = EEG_num_channels
    EMG_CH = EMG_num_channels
    EEG_CLASSES = 3
    EMG_CLASSES = 5
    TOTAL_CLASSES = EMG_CLASSES
    NUM_EPOCHS = 250                 # 150 - 200
    PATIENCE = 25                   # Early stopping patience - 25

    #===========#
    # Constants #
    #===========#
    global_best_vloss = float("inf")                # Used to only save one model.

    #====================================#
    # SHERPA Hyperparameter Optimazation #
    #====================================#

    parameters = model_handler_ins.get_hyperparameters()
    
    # algorithm = sherpa.algorithms.RandomSearch(max_num_trials = MAX_NUM_TRIALS)
    algorithm = sherpa.algorithms.GPyOpt(
        max_num_trials = MAX_NUM_TRIALS,
        acquisition_type = 'EI',                     # Expected improvement
        num_initial_data_points = NUM_INITIAL_DATA_POINTS                 # Number of hyperparameter configurations before model learns
    )
    # Study represents the hyperparameter optimization itself
    study = sherpa.Study(
        parameters = parameters,
        algorithm = algorithm,
        lower_is_better = True,
        disable_dashboard = True
    )

    for trial in study:
        model_config = model_handler_ins.build_model_config(
            trial,
            EEG_CH, EMG_CH,
            EEG_CLASSES, EMG_CLASSES,
            TOTAL_CLASSES,
            EEG_num_samples, EMG_num_samples
        )
        train_config = model_handler_ins.build_training_config(
            trial = trial
        )
        #=======================#
        # Multi fusion datasets #
        #=======================#
        model = model_handler_ins.get_model(config = model_config)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params = model.parameters(), lr = train_config['lr'], weight_decay = train_config['weight_decay'])

        # DataLoaders (update batch_size)
        train_loader = DataLoader(train_dataset_ins, batch_size = train_config['batch_size'], shuffle = True, pin_memory = pin_memory, num_workers = 0)
        val_loader = DataLoader(val_dataset_ins, batch_size = train_config['batch_size'], shuffle = False, pin_memory = pin_memory, num_workers = 0)
        test_loader = DataLoader(test_dataset_ins, batch_size = train_config['batch_size'], shuffle = False, pin_memory = pin_memory, num_workers = 0)

        best_train_loss = None
        best_val_loss = float("inf")
        best_val_acc = None

        best_epoch = 0
        best_state_dict = None
        early_stopping_counter = 0

        #===============================================================#
        # NOTE: Tensorboard config                                      #
        #   Enable all if using tensorboard                             #
        #   Change log_dir -> log_folder when saving model : torch.save #
        #===============================================================#
        # log_folder = os.path.join(log_dir, f"trial_{trial.id}")               
        # os.makedirs(log_folder, exist_ok=False)
        # writer = SummaryWriter(os.path.join(log_folder, 'trial_{}_timestamp_{}'.format(trial.id, timestamp)))

        for epoch in range(NUM_EPOCHS):

            # Train model
            avg_train_loss = train_eval_ins.train_one_epoch(model = model, train_loader = train_loader, criterion = criterion, optimizer = optimizer, device = device)

            # Validate model
            avg_vloss, vacc, _ = train_eval_ins.validation_one_epoch(model = model, val_loader = val_loader, criterion = criterion, device = device)
            
            # Tensor Board logging
            # writer.add_scalars('Loss', { 'Training' : avg_train_loss, 'Validation' : avg_vloss }, epoch + 1)
            # writer.add_scalars('Accuracy Validation', {'Validation' : vacc }, epoch + 1)
            # writer.flush()

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

        if best_val_loss < global_best_vloss :
            global_best_vloss = best_val_loss
            
            torch.save({
                "model_name": model_name,
                "model_state": best_state_dict,
                "model_args": model_config, 
                "optimizer_state_dict": best_optimizer_dict,
                "hyperparameters": trial.parameters,}, 
                r'{}\model.pth'.format(log_dir))

        # writer.close()
        study.finalize(trial, status = 'COMPLETED')

#==================================#
# Traning of model across subjects #
#==================================#
def singleNet_classfication_acrossSubjects(subject_name : str | list, sherpa_log_folder : str = 'SingleNet_LSTM_EMG', data_type : str = None):
    # When chancing between EEG and EMG
    # preprocessing instance
    # Load function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()              # Use pin_memory if CUDA is available
    print(f"Using device: {device}")
    print("Pin memory set to:", pin_memory)

    LOG_NAME = 'subject_0-2'
    log_dir = Path(__file__).resolve().parent / f'loggings/{sherpa_log_folder}/{LOG_NAME}'         # Path(__file__).resolve() -> Absolute path to this file
    data_dir = Path(__file__).resolve().parents[2] / 'src/experiment/data'
    data_type = str.upper(data_type)
    #==========================#
    # NOTE: Tensorboard config #
    #==========================#
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')                                        # Use when having tensorboard
    os.makedirs(log_dir, exist_ok=False)                                                          # use without tensorboard

    logger_ins = ExperimentLogger(save_path = log_dir)
    load_ins = load_datasets(base_dir = data_dir)
    split_ins = Manage3Split(seed = SEED)
    train_eval_ins = SingleNet_train_eval()

    if data_type == 'EMG':
        EMG_ins = EMG_preprocessing(fs = EMG_FREQ, bandpass_lowcut = EMG_LOWCUT, bandpass_highcut = EMG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    elif data_type == 'EEG':
        EEG_ins = EEG_preprocessing(fs = EEG_FREQ, bandpass_lowcut = EEG_LOWCUT, bandpass_highcut = EEG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    else:
        raise ValueError('data_type must be either EMG or EEG')

    #====================#
    # Load Training data #
    #====================#
    X_train_index = []
    X_train_thumb = []

    for subj in subject_name:
        if data_type == 'EMG':
            X_epoch_index, _, _ = load_ins.load_EMG_data(subject_name = subj, finger_name = 'index', EMG_config_dict = EMG_CONFIG_DICT, reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EMG_ins.preprocessing_routine)
            X_epoch_thumb, _, _ = load_ins.load_EMG_data(subject_name = subj, finger_name = 'thumb', EMG_config_dict = EMG_CONFIG_DICT, reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EMG_ins.preprocessing_routine)
        else:
            X_epoch_index, _ = load_ins.load_EEG_data(subject_name = subj, finger_name = 'index', reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EEG_ins.preprocessing_routine, EEG_useable_channels = EEG_USEABLE_CHANNELS)
            X_epoch_thumb, _ = load_ins.load_EEG_data(subject_name = subj, finger_name = 'thumb', reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EEG_ins.preprocessing_routine, EEG_useable_channels = EEG_USEABLE_CHANNELS)

        if subj == 'subject_2':
            X_test_index = X_epoch_index
            X_test_thumb = X_epoch_thumb
        else:
            X_train_index.append(X_epoch_index)
            X_train_thumb.append(X_epoch_thumb)
    
    X_train_index = np.concatenate(X_train_index, axis = 0)
    X_train_thumb = np.concatenate(X_train_thumb, axis = 0)

    FREQ = RMS_FREQ if data_type == 'EMG' else EEG_FREQ

    # Slice X_test_... into validation/test split
    X_test_num_epochs = X_test_index.shape[0]
    X_val_slice_index = X_test_index[:X_test_num_epochs//2]          # For validation
    X_test_slice_index = X_test_index[X_test_num_epochs//2:]          # For testing

    X_test_num_epochs = X_test_thumb.shape[0]
    X_val_slice_thumb = X_test_thumb[:X_test_num_epochs//2]          # For validation
    X_test_slice_thumb = X_test_thumb[X_test_num_epochs//2:]          # For testing

    X_train, y_train = split_ins._build_split(epoch_index = X_train_index,
                                              epoch_thumb = X_train_thumb,
                                              index_trials_indices = slice(None),
                                              thumb_trials_indices = slice(None),
                                              fs = FREQ)

    X_val, y_val = split_ins._build_split(epoch_index = X_val_slice_index,
                                          epoch_thumb = X_val_slice_thumb,
                                          index_trials_indices = slice(None),
                                          thumb_trials_indices = slice(None),
                                          fs = FREQ)
    
    X_test, y_test = split_ins._build_split(epoch_index = X_test_slice_index,
                                            epoch_thumb = X_test_slice_thumb,
                                            index_trials_indices = slice(None),
                                            thumb_trials_indices = slice(None),
                                            fs = FREQ)
        
    _, num_samples, num_channels = X_train.shape

    #=================#
    # Single datasets #
    #=================#
    print('\nTraining dataset shapes:')
    train_dataset_ins = SingleManageDataset(X_train, y_train, data_type = data_type)
    print('Validation dataset shapes:')
    val_dataset_ins = SingleManageDataset(X_val, y_val, data_type = data_type)
    print('Testing dataset shapes:')
    test_dataset_ins = SingleManageDataset(X_test, y_test, data_type = data_type)

    #========================================================#
    # THESE PARAMETERS ARE CHANCEABLE, DEPENDING ON THE TASK #
    #========================================================#
    MAX_NUM_TRIALS = 100             # 75 - 250 (simply to max) 
    DATA_CH = num_channels
    NUM_CLASSES = 5 if data_type == 'EMG' else 3
    NUM_EPOCHS = 250                 # 150 - 200
    PATIENCE = 25                   # Early stopping patience - 25
    NUM_INITIAL_DATA_POINTS = 20
    
    #===========#
    # Constants #
    #===========#
    global_best_vloss = float("inf")                # Used to only save one model.

    #====================================#
    # SHERPA Hyperparameter Optimazation #
    #====================================#

    parameters = [
        # General
        sherpa.Continuous(name='learning_rate', range=[0.00001, 0.001], scale='log'),
        sherpa.Continuous(name="weight_decay", range=[1e-6, 1e-2], scale="log"),  
        sherpa.Continuous(name='dropout', range=[0.1, 0.5]),
        sherpa.Ordinal(name='batch_size', range=[16, 32, 64]),
        sherpa.Ordinal(name='dense_ratio', range=[0.25, 0.5, 0.75, 1.0]),
        sherpa.Choice(name='activation', range=['relu', 'elu']),

        # LSTM
        sherpa.Ordinal(name='num_hidden_units', range=[32, 64]),          # 32, 64, 128, 256          
        sherpa.Choice(name="bidirectional", range=[False, True]),                      
        sherpa.Choice(name='lstm_layers', range=[1, 2, 3]),

        # CNN
        # sherpa.Ordinal(name='cnn_filters', range=[16, 32, 64]),
        # sherpa.Ordinal(name='EEG_kernel_ratio', range=[0.01, 0.02, 0.03, 0.04, 0.1]),   # EEG [3.75, 7.5, 11.25, 15, 37.5] samples
        # sherpa.Ordinal(name='EMG_kernel_ratio', range=[0.03, 0.06, 0.09, 0.12, 0.15]),   # EEG : 3.6, 7.2, 10.8, 14.4, 18
    ]
    
    # algorithm = sherpa.algorithms.RandomSearch(max_num_trials = MAX_NUM_TRIALS)
    algorithm = sherpa.algorithms.GPyOpt(
        max_num_trials = MAX_NUM_TRIALS,
        acquisition_type = 'EI',                     # Expected improvement
        num_initial_data_points = NUM_INITIAL_DATA_POINTS                 # Number of hyperparameter configurations before model learns
    )
    # Study represents the hyperparameter optimization itself
    study = sherpa.Study(
        parameters = parameters,
        algorithm = algorithm,
        lower_is_better = True,
        disable_dashboard = True
    )

    for trial in study:
        # General
        dropout = trial.parameters['dropout']       
        batch_size = trial.parameters['batch_size']
        activation = trial.parameters['activation'] 
        dense_ratio = trial.parameters['dense_ratio']
        weight_decay = trial.parameters["weight_decay"]
        lr = trial.parameters["learning_rate"]

        # LSTM
        num_hidden_units = trial.parameters['num_hidden_units']
        lstm_layers = trial.parameters['lstm_layers']
        bidirectional = trial.parameters['bidirectional']   
        
        # CNN
        # cnn_filters = trial.parameters['cnn_filters']
        # kernel_ratio = trial.parameters['EMG_kernel_ratio']
        # kernel_size = kernel_from_ratio(seq_len = num_samples, ratio = kernel_ratio, min_kernel = 3)

        #=================#
        # Single datasets #
        #=================#
        model = SingleNet_LSTM(input_dim = DATA_CH, output_dim = NUM_CLASSES, hidden_dim = num_hidden_units, lstm_layers = lstm_layers, bidirectional = bidirectional, dropout = dropout, activation = activation, dense_ratio = dense_ratio)
        # model = SingleNet_CNN_LSTM(input_dim = DATA_CH,
        #                            output_dim = NUM_CLASSES,
        #                            hidden_dim = num_hidden_units,
        #                            lstm_layers = lstm_layers,
        #                            bidirectional = bidirectional,
        #                            dropout = dropout,
        #                            activation = activation,
        #                            dense_ratio = dense_ratio,
        #                            cnn_filters = cnn_filters,
        #                            kernel_size = kernel_size)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params = model.parameters(), lr = lr, weight_decay = weight_decay)

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

        #===================================#
        # NOTE: Tensorboard config          #
        #   Enable all if using tensorboard #
        #===================================#
        # log_folder = os.path.join(log_dir, f"trial_{trial.id}")               
        # os.makedirs(log_folder, exist_ok=False)
        # writer = SummaryWriter(os.path.join(log_folder, 'trial_{}_timestamp_{}'.format(trial.id, timestamp)))

        for epoch in range(NUM_EPOCHS):

            # Train model
            avg_train_loss = train_eval_ins.train_one_epoch(model = model, train_loader = train_loader, criterion = criterion, optimizer = optimizer, device = device)

            # Validate model
            avg_vloss, vacc, _ = train_eval_ins.validaton_one_epoch(model = model, val_loader = val_loader, criterion = criterion, device = device)

            # Tensor Board logging
            # writer.add_scalars('Loss', { 'Training' : avg_train_loss, 'Validation' : avg_vloss }, epoch + 1)
            # writer.add_scalars('Accuracy Validation', {'Validation' : vacc }, epoch + 1)
            # writer.flush()

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

        if best_val_loss < global_best_vloss :
            global_best_vloss = best_val_loss

          
            model_arg = {
            "input_dim": DATA_CH,
            "output_dim" : NUM_CLASSES,
            "hidden_dim": num_hidden_units,
            "lstm_layers": lstm_layers,
            "bidirectional" : bidirectional,
            "dropout": dropout,
            "activation": activation,
            "dense_ratio": dense_ratio,}
            # "cnn_filters" : cnn_filters,
            # "kernel_size" : kernel_size,
            
            torch.save({
                "model_state": best_state_dict,
                "model_args": model_arg, 
                "optimizer_state_dict": best_optimizer_dict,
                "hyperparameters": trial.parameters,}, 
                r'{}\model.pth'.format(log_dir))
            
        # writer.close()                            # NOTE: Enable with tensorboard
        study.finalize(trial, status = 'COMPLETED')

def fusionNet_classfication_acrossSubjects(subject_name : list, sherpa_log_folder : str = 'SingleNet_LSTM_EMG', model_name : str = 'FusionNet_LSTM'):
    '''
    Train a model with EEG and EMG across subjects. 
    Subjects are clearly separated between traning, validation and test split
    
    Parameters
    -----------
    subject_name : list
        List of all subjects to be included
    sherpa_log_folder : str
        Path to where the model and loggings need to be saved in the 'src/experiment/data' directory
    model_name : str
        Select between models.\n
        Options:
        1) FusionNet_LSTM
        2) FusionNet_CNN_LSTM
        3) FusionNet_CNN_LSTM_ATTENSION
    '''
    # When chancing between EEG and EMG
    # preprocessing instance
    # Load function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()              # Use pin_memory if CUDA is available
    print(f"Using device: {device}")
    print("Pin memory set to:", pin_memory)

    LOG_NAME = 'subject_0-2_test'
    log_dir = Path(__file__).resolve().parent / f'loggings/{sherpa_log_folder}/{LOG_NAME}'         # Path(__file__).resolve() -> Absolute path to this file
    data_dir = Path(__file__).resolve().parents[2] / 'src/experiment/data'

    #==========================#
    # NOTE: Tensorboard config #
    #==========================#
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')                                        # Use when having tensorboard
    os.makedirs(log_dir, exist_ok=False)                                                          # use without tensorboard

    logger_ins = ExperimentLogger(save_path = log_dir)
    load_ins = load_datasets(base_dir = data_dir)
    split_ins = Manage3Split(seed = SEED)
    EMG_ins = EMG_preprocessing(fs = EMG_FREQ, bandpass_lowcut = EMG_LOWCUT, bandpass_highcut = EMG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    EEG_ins = EEG_preprocessing(fs = EEG_FREQ, bandpass_lowcut = EEG_LOWCUT, bandpass_highcut = EEG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    train_eval_ins = FusionNet_train_eval()
    model_handler_ins = FusionNetHandler(model_name = model_name)
    #====================#
    # Load Training data #
    #====================#
    EEG_train_index = []
    EEG_train_thumb = []
    EMG_train_index = []
    EMG_train_thumb = []

    for subj in subject_name:
        EEG_epoch_index, EMG_epoch_index, _, _ = load_ins.load_EEG_EMG_data(subject_name = subj, finger_name = 'index', reject_config_dict = REJECT_CONFIG_DICT, EEG_preprocessing_func = EEG_ins.preprocessing_routine, EMG_preprocessing_func = EMG_ins.preprocessing_routine, EMG_config_dict = EMG_CONFIG_DICT, EEG_useable_channels = EEG_USEABLE_CHANNELS)
        EEG_epoch_thumb, EMG_epoch_thumb, _, _ = load_ins.load_EEG_EMG_data(subject_name = subj, finger_name = 'thumb', reject_config_dict = REJECT_CONFIG_DICT, EEG_preprocessing_func = EEG_ins.preprocessing_routine, EMG_preprocessing_func = EMG_ins.preprocessing_routine, EMG_config_dict = EMG_CONFIG_DICT, EEG_useable_channels = EEG_USEABLE_CHANNELS)
       
        if subj == 'subject_2':
            EEG_test_index, EMG_test_index = EEG_epoch_index, EMG_epoch_index
            EEG_test_thumb, EMG_test_thumb = EEG_epoch_thumb, EMG_epoch_thumb
        else:
            EEG_train_index.append(EEG_epoch_index)
            EEG_train_thumb.append(EEG_epoch_thumb)
            EMG_train_index.append(EMG_epoch_index)
            EMG_train_thumb.append(EMG_epoch_thumb)
    
    # Across multiple subjects
    EEG_train_index = np.concatenate(EEG_train_index, axis = 0)
    EEG_train_thumb = np.concatenate(EEG_train_thumb, axis = 0)
    EMG_train_index = np.concatenate(EMG_train_index, axis = 0)
    EMG_train_thumb = np.concatenate(EMG_train_thumb, axis = 0)

    # Slice X_test_... into validation/test split
    def _validation_test_split(data):
        num_epochs = data.shape[0]
        halfway = num_epochs // 2

        val_split = data[:halfway]
        test_split = data[halfway:]

        return val_split, test_split
    
    def _build_split(train_index, train_thumb, val_index, val_thumb, test_index, test_thumb, fs):

        X_train, y_train = split_ins._build_split(epoch_index = train_index,
                                                epoch_thumb = train_thumb,
                                                index_trials_indices = slice(None),
                                                thumb_trials_indices = slice(None),
                                                fs = fs)

        X_val, y_val = split_ins._build_split(epoch_index = val_index,
                                            epoch_thumb = val_thumb,
                                            index_trials_indices = slice(None),
                                            thumb_trials_indices = slice(None),
                                            fs = fs)
        
        X_test, y_test = split_ins._build_split(epoch_index = test_index,
                                                epoch_thumb = test_thumb,
                                                index_trials_indices = slice(None),
                                                thumb_trials_indices = slice(None),
                                                fs = fs)

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    # Split EEG dataset into 50% validation and 50% test datasets
    EEG_val_index_split, EEG_test_index_split = _validation_test_split(data = EEG_test_index)
    EEG_val_thumb_split, EEG_test_thumb_split = _validation_test_split(data = EEG_test_thumb)
    # Split EMG dataset into 50% validation and 50% test datasets
    EMG_val_index_split, EMG_test_index_split = _validation_test_split(data = EMG_test_index)
    EMG_val_thumb_split, EMG_test_thumb_split = _validation_test_split(data = EMG_test_thumb)

    X_EEG_train, X_EEG_val, X_EEG_test, y_EEG_train, y_EEG_val, y_EEG_test = _build_split(train_index = EEG_train_index,
                                                                              train_thumb = EEG_train_thumb,
                                                                              val_index = EEG_val_index_split,
                                                                              val_thumb = EEG_val_thumb_split,
                                                                              test_index = EEG_test_index_split,
                                                                              test_thumb = EEG_test_thumb_split,
                                                                              fs = EEG_FREQ)
    
    X_EMG_train, X_EMG_val, X_EMG_test, y_EMG_train, y_EMG_val, y_EMG_test = _build_split(train_index = EMG_train_index,
                                                                              train_thumb = EMG_train_thumb,
                                                                              val_index = EMG_val_index_split,
                                                                              val_thumb = EMG_val_thumb_split,
                                                                              test_index = EMG_test_index_split,
                                                                              test_thumb = EMG_test_thumb_split,
                                                                              fs = RMS_FREQ)
    
    _, EEG_num_samples, EEG_num_channels = X_EEG_train.shape
    _, EMG_num_samples, EMG_num_channels = X_EMG_train.shape

    #=======================#
    # Multi fusion datasets #
    #=======================#
    print('\nTraining dataset shapes:')
    train_dataset_ins = MultiManageDataset(X_EEG_train, X_EMG_train, y_EEG_train, y_EMG_train)
    print('Validation dataset shapes:')
    val_dataset_ins = MultiManageDataset(X_EEG_val, X_EMG_val, y_EEG_val, y_EMG_val)
    print('Testing dataset shapes:')
    test_dataset_ins = MultiManageDataset(X_EEG_test, X_EMG_test, y_EEG_test, y_EMG_test)

    #========================================================#
    # THESE PARAMETERS ARE CHANCEABLE, DEPENDING ON THE TASK #
    #========================================================#
    MAX_NUM_TRIALS = 100             # 75 - 250 (simply to max) 
    NUM_INITIAL_DATA_POINTS = 20
    EEG_CH = EEG_num_channels
    EMG_CH = EMG_num_channels
    EEG_CLASSES = 3
    EMG_CLASSES = 5
    TOTAL_CLASSES = EMG_CLASSES
    NUM_EPOCHS = 250                 # 150 - 200
    PATIENCE = 25                   # Early stopping patience - 25
    
    #===========#
    # Constants #
    #===========#
    global_best_vloss = float("inf")                # Used to only save one model.

    #====================================#
    # SHERPA Hyperparameter Optimazation #
    #====================================#

    parameters = model_handler_ins.get_hyperparameters()
    
    # algorithm = sherpa.algorithms.RandomSearch(max_num_trials = MAX_NUM_TRIALS)
    algorithm = sherpa.algorithms.GPyOpt(
        max_num_trials = MAX_NUM_TRIALS,
        acquisition_type = 'EI',                     # Expected improvement
        num_initial_data_points = NUM_INITIAL_DATA_POINTS                 # Number of hyperparameter configurations before model learns
    )
    # Study represents the hyperparameter optimization itself
    study = sherpa.Study(
        parameters = parameters,
        algorithm = algorithm,
        lower_is_better = True,
        disable_dashboard = True
    )

    for trial in study:
        model_config = model_handler_ins.build_model_config(
            trial,
            EEG_CH, EMG_CH,
            EEG_CLASSES, EMG_CLASSES,
            TOTAL_CLASSES,
            EEG_num_samples, EMG_num_samples
        )
        train_config = model_handler_ins.build_training_config(
            trial = trial
        )
        print(model_config)
        print(train_config)
        #=======================#
        # Multi fusion datasets #
        #=======================#
        model = model_handler_ins.get_model(config = model_config)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params = model.parameters(), lr = train_config['lr'], weight_decay = train_config['weight_decay'])

        # DataLoaders (update batch_size)
        train_loader = DataLoader(train_dataset_ins, batch_size = train_config['batch_size'], shuffle = True, pin_memory = pin_memory, num_workers = 0)
        val_loader = DataLoader(val_dataset_ins, batch_size = train_config['batch_size'], shuffle = False, pin_memory = pin_memory, num_workers = 0)
        test_loader = DataLoader(test_dataset_ins, batch_size = train_config['batch_size'], shuffle = False, pin_memory = pin_memory, num_workers = 0)
        exit()
        best_train_loss = None
        best_val_loss = float("inf")
        best_val_acc = None

        best_epoch = 0
        best_state_dict = None
        early_stopping_counter = 0

        #===================================#
        # NOTE: Tensorboard config          #
        #   Enable all if using tensorboard #
        #===================================#
        # log_folder = os.path.join(log_dir, f"trial_{trial.id}")               
        # os.makedirs(log_folder, exist_ok=False)
        # writer = SummaryWriter(os.path.join(log_folder, 'trial_{}_timestamp_{}'.format(trial.id, timestamp)))

        for epoch in range(NUM_EPOCHS):

            # Train model
            avg_train_loss = train_eval_ins.train_one_epoch(model = model, train_loader = train_loader, criterion = criterion, optimizer = optimizer, device = device)

            # Validate model
            avg_vloss, vacc, _ = train_eval_ins.validation_one_epoch(model = model, val_loader = val_loader, criterion = criterion, device = device)

            # Tensor Board logging
            # writer.add_scalars('Loss', { 'Training' : avg_train_loss, 'Validation' : avg_vloss }, epoch + 1)
            # writer.add_scalars('Accuracy Validation', {'Validation' : vacc }, epoch + 1)
            # writer.flush()

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

        if best_val_loss < global_best_vloss :
            global_best_vloss = best_val_loss

            torch.save({
                "model_name": model_name,
                "model_state": best_state_dict,
                "model_args": model_config, 
                "optimizer_state_dict": best_optimizer_dict,
                "hyperparameters": trial.parameters,}, 
                r'{}\model.pth'.format(log_dir))
            
        # writer.close()                            # NOTE: Enable with tensorboard
        study.finalize(trial, status = 'COMPLETED')

#================#
# Analyse models #
#================#
def inspect_model(subject_name = 'subject_0', sherpa_log_folder = 'SingleNet_LSTM_EMG'):
    # sherpa_info_path = Path(__file__).resolve().parent / f"loggings/{logging_name}/SHERPA_results.pt"

            
    sherpa_info_path = Path(__file__).resolve().parent / f"loggings/{sherpa_log_folder}/{subject_name}/SHERPA_results.pt"
    # sherpa_info_path = Path(__file__).resolve().parent / f"loggings/SHERPA_results.pt"
    if not os.path.exists(sherpa_info_path):
        raise FileExistsError(sherpa_info_path)
    data = torch.load(sherpa_info_path, weights_only=False)

    # Extract test accuracies
    # Extract test accuracies
    '''acc_list = [trial['validation_loss'] for trial in data['trials']]

    # Get indices sorted from highest → lowest accuracy
    sorted_indices = sorted(range(len(acc_list)), key=lambda i: acc_list[i], reverse=False)

    # Iterate over trials in sorted order
    for rank, idx in enumerate(sorted_indices):
        trial = data['trials'][idx]

        print('Rank:', rank + 1)
        print('Trial:', idx + 1)
        print('Epochs', trial['best_epoch'])
        print('Training loss:' , trial['training_loss'])
        print('Validation loss:', trial['validation_loss'])
        print('Validation accuracy:', trial['validation_accuracy'])
        print('Test accuracy:', trial['test_accuracy'])
        print('Hyperparameters:\n', trial['hyperparameters'], '\n')
        if rank > 10:
            break'''
    
    print('Last ten')
    data_len = len(data['trials'])
    for idx in range(data_len - 50, data_len):
        trial = data['trials'][idx]

        # print('Rank:', rank + 1)
        print('Trial:', idx + 1)
        print('Epochs', trial['best_epoch'])
        print('Training loss:' , trial['training_loss'])
        print('Validation loss:', trial['validation_loss'])
        print('Validation accuracy:', trial['validation_accuracy'])
        print('Test accuracy:', trial['test_accuracy'])
        print('Hyperparameters:\n', trial['hyperparameters'], '\n')

    best_vloss = min(
        data["trials"],
        key=lambda x: x["validation_loss"]
    )
    best_tacc = max(
        data['trials'],
        key = lambda x: x['test_accuracy']
    )

    print(f'\n---------{subject_name}-----------')
    for best_name, best_value in zip(['lowest validation loss', 'highest test accuracy'], [best_vloss, best_tacc]):
        cm = confusion_matrix(best_value['labels'], best_value['predictions']) 
        print(f'For {best_name}')
        print('         Best trial ID: ', best_value['trial_id'])
        print('         Stoped at epoch', best_value['best_epoch'])
        print('         Training loss:' , best_value['training_loss'])
        print('         validation loss', best_value['validation_loss'])
        print('         Test accuracy: ', best_value["test_accuracy"])
        print('         Hyperparameter: ', best_value['hyperparameters'])
        print(cm)
        print('\n')

def singleNet_inspect_model(subject_name = 'subject_0', sherpa_log_folder = 'SingleNet_LSTM_EMG'):
    # model_path_folder = Path(__file__).resolve().parent / f"loggings/{sherpa_log_folder}/{subject_name}"
    # sherpa_info_path = model_path_folder / 'SHERPA_results.pt'

    model_path_folder = Path(__file__).resolve().parent / f"sherpa_loggings/{sherpa_log_folder}"
    sherpa_info_path = model_path_folder / f'{subject_name}_SHERPA_results.pt'

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
    print(f'\n---------{subject_name}-----------')
    print('Best trial ID: ', best_trial_id)
    print('Stoped at epoch', best['best_epoch'])
    print('Training loss:' , best['training_loss'])
    print('validation loss', best['validation_loss'])
    print('Test accuracy: ', best["test_accuracy"])
    print('Hyperparameters :\n', best['hyperparameters'])
    print('\n')
    
    model_path = model_path_folder / 'model.pth'
    checkpoint = torch.load(f = model_path, map_location = device)

    print(checkpoint["model_args"])
    model_args = checkpoint["model_args"]

    # Add missing arguments
    
    # model_args['cnn_filters'] = best['hyperparameters']['cnn_filters']
    # model_args['kernel_size'] = best['hyperparameters']['kernel_size']
    # model_args["eeg_output_dim"] = 3
    # model_args["emg_output_dim"] = 5
    # model_args["dense_fusion_layer"] = 16

    # model_interference = SingleNet_LSTM(**checkpoint["model_args"])
    model_interference = SingleNet_LSTM(**model_args)

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
    # X_epoch_index, _ = load_ins.load_EEG_data(subject_name = subject_name, finger_name = 'index', reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EEG_ins.preprocessing_routine, EEG_useable_channels = EEG_USEABLE_CHANNELS)
    # X_epoch_thumb, _ = load_ins.load_EEG_data(subject_name = subject_name, finger_name = 'thumb', reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EEG_ins.preprocessing_routine, EEG_useable_channels = EEG_USEABLE_CHANNELS)

    FREQ = RMS_FREQ

    num_index_trials = X_epoch_index.shape[0]
    num_thumb_trials = X_epoch_thumb.shape[0]

    _, _, X_test, _, _, y_test = split_ins.build_modality_split(
        num_index_trials = num_index_trials,
        num_thumb_trials = num_thumb_trials,
        epoch_index = X_epoch_index,
        epoch_thumb = X_epoch_thumb,
        fs = FREQ)
    
    #=================#
    # Single datasets #
    #=================#
    print('Testing dataset shapes:')
    test_dataset_ins = SingleManageDataset(X_test, y_test, data_type = 'EMG')

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

    model_interference.eval()

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

    _plot_tsne_context(context = all_logits, labels = y_test, perplexity = 40, n_iter = 1000, random_state = 42)
    # plot_tsne_context(context = all_context, labels = y_test, perplexity = 40, n_iter = 1000, random_state = 42)

    score = silhouette_score(all_logits, y_test)
    print(score)

    # ~0.5 → good separation
    # ~0.2 → weak separation
    # ~0 → no separation
    # <0 → overlapping

def fusionNet_inspect_model(subject_name = 'subject_0', sherpa_log_folder = 'SingleNet_LSTM_EMG'):
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
    print(f'\n---------{subject_name}-----------')
    print('Best trial ID: ', best_trial_id)
    print('Stoped at epoch', best['best_epoch'])
    print('Training loss:' , best['training_loss'])
    print('validation loss', best['validation_loss'])
    print('Test accuracy: ', best["test_accuracy"])
    print('\n')
    
    model_path = model_path_folder / 'model.pth'
    checkpoint = torch.load(f = model_path, map_location = device)

    model_args = checkpoint["model_args"]

    # Add missing arguments
    # model_args["eeg_output_dim"] = 3
    # model_args["emg_output_dim"] = 5
    # model_args["dense_fusion_layer"] = 16

    model_interference = FusionNet_CNN_LSTM(**model_args)             # NOTE : **checkpoint["model_args"]
    
    model_interference.load_state_dict(checkpoint["model_state"])
    model_interference.to(device)

    #===================#
    # Load Test dataset #
    #===================#
    data_dir = Path(__file__).resolve().parents[2] / 'src/experiment/data'
    
    load_ins = load_datasets(base_dir = data_dir)
    split_ins = Manage3Split(seed = SEED)
    EMG_ins = EMG_preprocessing(fs = EMG_FREQ, bandpass_lowcut = EMG_LOWCUT, bandpass_highcut = EMG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    EEG_ins = EEG_preprocessing(fs = EEG_FREQ, bandpass_lowcut = EEG_LOWCUT, bandpass_highcut = EEG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)

    #===========#
    # Load data #
    #===========#
    EEG_epoch_index, EMG_epoch_index, _, _ = load_ins.load_EEG_EMG_data(subject_name = subject_name, finger_name = 'index', reject_config_dict = REJECT_CONFIG_DICT, EEG_preprocessing_func = EEG_ins.preprocessing_routine, EMG_preprocessing_func = EMG_ins.preprocessing_routine, EMG_config_dict = EMG_CONFIG_DICT, EEG_useable_channels = EEG_USEABLE_CHANNELS)
    EEG_epoch_thumb, EMG_epoch_thumb, _, _ = load_ins.load_EEG_EMG_data(subject_name = subject_name, finger_name = 'thumb', reject_config_dict = REJECT_CONFIG_DICT, EEG_preprocessing_func = EEG_ins.preprocessing_routine, EMG_preprocessing_func = EMG_ins.preprocessing_routine, EMG_config_dict = EMG_CONFIG_DICT, EEG_useable_channels = EEG_USEABLE_CHANNELS)

    num_index_trials = EEG_epoch_index.shape[0]
    num_thumb_trials = EEG_epoch_thumb.shape[0]

    X_EEG_train, X_EEG_val, X_EEG_test, y_EEG_train, y_EEG_val, y_EEG_test = split_ins.build_modality_split(
        num_index_trials = num_index_trials,
        num_thumb_trials = num_thumb_trials,
        epoch_index = EEG_epoch_index,
        epoch_thumb = EEG_epoch_thumb,
        fs = EEG_FREQ
    )

    X_EMG_train, X_EMG_val, X_EMG_test, y_EMG_train, y_EMG_val, y_EMG_test = split_ins.build_modality_split(
        num_index_trials = num_index_trials,
        num_thumb_trials = num_thumb_trials,
        epoch_index = EMG_epoch_index,
        epoch_thumb = EMG_epoch_thumb,
        fs = RMS_FREQ
    )

    
    #=================#
    # Single datasets #
    #=================#
    print('Testing dataset shapes:')
    test_dataset_ins = MultiManageDataset(X_EEG_test, X_EMG_test, y_EEG_test, y_EMG_test)

    test_loader = DataLoader(test_dataset_ins, batch_size = batch_size, shuffle = False, pin_memory = pin_memory, num_workers = 0)

    #===================#
    # Perform inference #
    #===================#

    correct_fusion = 0
    total = 0

    all_preds = []
    all_labels = []
    all_logits = []
    # all_context = []

    model_interference.eval()

    criterion = nn.CrossEntropyLoss()
    loss_eeg_all = 0
    loss_emg_all = 0
    loss_final_all = 0
    loss_all = 0
    correct_eeg = 0
    correct_emg = 0
    

    with torch.no_grad():
        for eeg, emg, eeg_lab, emg_lab in test_loader:
            X_eeg, X_emg, y_eeg, y_emg = eeg.to(device), emg.to(device), eeg_lab.to(device), emg_lab.to(device)

            # Forward pass
            final_logits, eeg_logits, emg_logits, _, _ = model_interference(eeg = X_eeg, emg = X_emg)
            
            # Predicted class index
            _, fusion_pred = torch.max(final_logits, dim=1)             # index of max value (predicted class)

            # Compute the loss and its gradients
            loss_final = criterion(final_logits, y_emg)        # EMG has all 5 lables (contract per finger, release per finger, rest all fingers)
            loss_eeg   = criterion(eeg_logits, y_eeg)          # EEG has only 3 lables (contract, release, rest)
            loss_emg   = criterion(emg_logits, y_emg)  

            loss = loss_final + 0.3 * loss_eeg + 0.3 * loss_emg     # Only used for training optimization

            # Accuracy statistics
            total += y_emg.size(0)
            correct_fusion += (fusion_pred == y_emg).sum().item()
            
            # Store outputs for confusion matrix etc.
            all_preds.append(fusion_pred.cpu())
            all_labels.append(y_emg.cpu())
            all_logits.append(final_logits.cpu())
            # all_context.append(context.cpu())

            #=====================================#
            # Inspect contribution of EEG and EMG #
            #=====================================#
            eeg_pred = torch.argmax(eeg_logits, dim=1)
            emg_pred = torch.argmax(emg_logits, dim=1)
            correct_eeg += (eeg_pred == y_eeg).sum().item()
            correct_emg += (emg_pred == y_emg).sum().item()
            loss_final_all += loss_final.item()
            loss_eeg_all += loss_eeg.item()
            loss_emg_all += loss_emg.item()
            loss_all += loss.item()

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_logits = torch.cat(all_logits).numpy()
        # all_context = torch.cat(all_context).numpy()

        num_batches = len(test_loader)
        print('\n------LOSSES-------\n')
        print(f"Avg EEG loss   : {loss_eeg_all / num_batches :.4f}")
        print(f"Avg EMG loss   : {loss_emg_all / num_batches :.4f}")
        print(f"Avg final loss : {loss_final_all / num_batches :.4f}")
        print(f"Avg comb loss  : {loss_all / num_batches :.4f}")
        print('\n-------Accuracies---------\n')
        print(f'EEG accuracy : {(correct_eeg / total) * 100 :.2f}')
        print(f'EMG accuracy : {(correct_emg / total) * 100 :.2f}')
        print(f"Fusion accuracy: {(correct_fusion / total) * 100 :.2f}")

        #==========#
        # Analysis #
        #==========#

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)

        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        print(cm_norm)
    
    _plot_tsne_context(context = all_logits, labels = y_EMG_test, perplexity = 40, n_iter = 1000, random_state = 42)
    # plot_tsne_context(context = all_context, labels = y_test, perplexity = 40, n_iter = 1000, random_state = 42)

    score = silhouette_score(all_logits, y_EMG_test)
    print(score)

    # ~0.5 → good separation
    # ~0.2 → weak separation
    # ~0 → no separation
    # <0 → overlapping

def _plot_tsne_context(
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
        random_state=random_state,
        n_jobs=1
    )

    X_embedded = tsne.fit_transform(X)

    import matplotlib.colors as mcolors

    # ---- convert labels to integer classes ----
    # Example: map unique labels to 0..4
    unique_classes = np.unique(y)
    class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
    y_int = np.vectorize(class_mapping.get)(y)

    num_classes = len(unique_classes)

    # ---- discrete colormap ----
    cmap = plt.get_cmap("tab10", num_classes)
    norm = mcolors.BoundaryNorm(
        boundaries=np.arange(-0.5, num_classes + 0.5, 1),
        ncolors=num_classes
    )

    # ---- plot ----
    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=y_int,
        cmap=cmap,
        norm=norm,
        alpha=0.7,
        s=25
    )
    class_names = ['Index contract', 'Index release', 'Thumb contract', 'Thumb release', 'Rest']
    cbar = plt.colorbar(scatter, ticks=range(num_classes), label="Class")
    cbar.ax.set_yticklabels(class_names)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_model_groups_subjects(
    subject_ids,
    eeg_acc,
    emg_acc,
    fusion_acc,
    title="Test Accuracy per Model and Subject",
    ylabel="Accuracy (%)"
):
    """
    Plot grouped bars where each group is a model (EEG, EMG, Fusion)
    and each bar inside the group represents a subject.

    Parameters
    ----------
    subject_ids : list
        Example: ['S1','S2','S3','S4']

    eeg_acc : list
        Accuracy per subject for EEG model

    emg_acc : list
        Accuracy per subject for EMG model

    fusion_acc : list
        Accuracy per subject for Fusion model
    """

    models = ['LSTM-EEG', 'LSTM-EMG', 'LSTM-Fusion', 'CNN+LSTM-EEG', 'CNN+LSTM-EMG', 'CNN+LSTM-fusion', 'CNN+LSTM+Attension-EEG', 'CNN+LSTM+Attension-EMG', 'CNN+LSTM+Attension-fusion']
    data = [eeg_acc, emg_acc, fusion_acc]

    n_models = len(models)
    n_subjects = len(subject_ids)

    bar_width = 0.5 / n_subjects
    x = np.arange(n_models)

    plt.figure(figsize=(10,6))

    for i, subject in enumerate(subject_ids):
        subject_values = [data[m][i] for m in range(n_models)]
        offset = (i - (n_subjects - 1)/2) * bar_width

        plt.bar(
            x + offset,
            subject_values,
            bar_width,
            label=subject
        )

    plt.xticks(x, models)
    plt.ylabel(ylabel)
    plt.xlabel("Model Type")
    plt.title(title)
    plt.legend(title="Subjects")
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()

def _plot_subject_accuracy_hierarchical(subject_ids, accuracies, architectures):
    """
    Plot grouped bars with hierarchical x-axis:
    Architecture -> Modality

    Parameters
    ----------
    subject_ids : list
        Example: ['S1','S2','S3']

    accuracies : dict
        Dictionary structured like:

        {
            "LSTM": {
                "EEG": [...],
                "EMG": [...],
                "Fusion": [...]
            },
            "CNN+LSTM": {
                "EEG": [...],
                "EMG": [...],
                "Fusion": [...]
            },
            "CNN+LSTM+Attention": {
                "EEG": [...],
                "EMG": [...],
                "Fusion": [...]
            }
        }

    architectures : list
        Example: ['LSTM','CNN+LSTM','CNN+LSTM+Attention']
    """

    modalities = ["EEG", "EMG", "Fusion"]

    n_subjects = len(subject_ids)
    bar_width = 0.5 / n_subjects

    x_positions = []
    x_labels = []

    # Build positions
    pos = 0
    arch_centers = []

    for arch in architectures:

        start = pos

        for mod in modalities:
            x_positions.append(pos)
            x_labels.append(mod)
            pos += 1

        end = pos - 1
        arch_centers.append((start + end) / 2)

        pos += 0.5  # spacing between architectures

    plt.figure(figsize=(12,6))

    # Plot bars per subject
    for i, subject in enumerate(subject_ids):

        offset = (i - (n_subjects - 1)/2) * bar_width

        values = []

        for arch in architectures:
            for mod in modalities:
                values.append(accuracies[arch][mod][i])

        plt.bar(
            np.array(x_positions) + offset,
            values,
            bar_width,
            label=subject
        )

    plt.xticks(x_positions, x_labels)
    plt.ylabel("Accuracy (%)")
    plt.yticks(np.arange(0, 100.1, 10))
    plt.ylim([0, 100])
    # plt.xlabel("Model / Modality")
    plt.legend(title="Subjects")

    # Add architecture labels
    for center, arch in zip(arch_centers, architectures):
        plt.text(center, -5, arch, ha='center', va='top', fontsize=11)

    plt.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()

def main():
    t0 = time.time()
    subjects = ['subject_0', 'subject_1', 'subject_2']

    # fusionNet_classfication(subject_name = subject, sherpa_log_folder = 'FusionNet_LSTM_fewerHyperparameters')
    # fusionNet_classfication_acrossSubjects(subject_name = subjects, sherpa_log_folder = 'FusionNet_LSTM_fewerHyperparameters', model_name = 'FusionNet_CNN_LSTM')
    singleNet_classfication_acrossSubjects(subject_name = subjects, sherpa_log_folder = 'SingleNet_LSTM_EEG', data_type='EEG')
    print('Classification COMPLETE\n'
          'Time it took: ', time.time() - t0, 's')

def summary_accuracies():
    subjects = ['S0','S1','S2']

    # Subjejcts 0-1 traning and subject 2 for test and validaiton
    #LSTM_fusion : 52.2 acc, 1.97 loss
    #LSTM_EMG : 56.86, 1.28 loss
    #LSTM_EEG : 50.93 acc, 0.84 loss

    #CNN+LSTM_fusion : 85.54 acc, 1.2 loss
    #CNN+LSTM_EMG : 88.62, 0.54 loss
    #CNN+LSTM_EEG : 47.9, 0.92 loss

    accuracies = {

    "LSTM":{
        "EEG":[59.3, 41.7, 58.3],
        "EMG":[92.6, 82.7, 96.3],           # subj3 : 83.3 , subj4 : 93.6 , subj5 : 97.5
        "Fusion":[92.3, 80.2, 96.0]         # subj3 : 91.7 , subj4 : 79.5 , subj5 : 85.2 , subj6 : 59.7]
    },

    "CNN+LSTM":{
        "EEG":[75.3, 66.7, 75.0],
        "EMG":[100, 95, 100],
        "Fusion":[100, 96.3, 100]
    },

    "CNN+LSTM+Attention":{
        "EEG":[0, 0, 0],
        "EMG":[0, 0, 0],
        "Fusion":[0, 0, 0]
    }}

    architectures = ['LSTM','CNN+LSTM','CNN+LSTM+Attention']

    _plot_subject_accuracy_hierarchical(subjects, accuracies, architectures)

if __name__ == '__main__':
    # main()
    # fusionNet_inspect_model(subject_name = 'subject_0', sherpa_log_folder = 'FusionNet_LSTM_FH')
    # singleNet_inspect_model(subject_name = 'subject_0', sherpa_log_folder = 'SingleNet_CNN+LSTM_EMG')

    inspect_model(subject_name = 'subject_0-2', sherpa_log_folder = 'SingleNet_LSTM_EEG')

    # for subj in ['subject_3', 'subject_4', 'subject_5', 'subject_6']:
    #     inspect_model(subject_name = subj, sherpa_log_folder = 'FusionNet_LSTM_fewerHyperparameters')

    # summary_accuracies()
    