    MAX_NUM_TRIALS = 150             # 75 - 250 (simply to max) 
    EEG_CH = EEG_num_channels
    EMG_CH = EMG_num_channels
    EEG_CLASSES = 3
    EMG_CLASSES = 5
    TOTAL_CLASSES = EMG_CLASSES
    NUM_EPOCHS = 250                 # 150 - 200
    PATIENCE = 50                   # Early stopping patience - 25
    NUM_INITIAL_DATA_POINTS = 125   # MISTAKE

        sherpa.Continuous(name='learning_rate', range=[0.00001, 0.001], scale='log'),
        sherpa.Continuous(name="weight_decay", range=[1e-6, 1e-2], scale="log"),  
        sherpa.Continuous(name='dropout', range=[0.1, 0.5]),
        sherpa.Ordinal(name='batch_size', range=[16, 32, 64]),
        sherpa.Ordinal(name='dense_ratio', range=[0.25, 0.5, 0.75, 1.0]),
        sherpa.Choice(name='activation', range=['relu', 'elu']),

        # LSTM
        sherpa.Ordinal(name='num_hidden_units', range=[32, 64, 128, 256]),
        sherpa.Choice(name="bidirectional", range=[False, True]),                      
        sherpa.Choice(name='lstm_layers', range=[1, 2, 3]),