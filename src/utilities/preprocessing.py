from scipy import signal
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import resample

class Filtering:
    def __init__(self, fs=125.0):
        self.fs = fs
        self.a = None
        self.b = None
        self.sos = None
        self.band_lowcut = None
        self.band_highcut = None

    def butter_bandpass(self, data, lowcut = 5, highcut = 40, order = 4):
        """
        Perform butterworth bandpass filtering. \n
        Perform notch before butter_bandpass \n
        args: \n
            EEG_data: EEG channel data - Dimension should be (samples, channels) \n
            lowcut: Define lowcut frequency \n
            highcut: Define highcut frequency \n
            order: Define order for the filter
        """
        self.band_lowcut = lowcut
        self.band_highcut = highcut
        # Create a Butterworth bandpass filter - b is the numerator coefficients, a is the denominator coefficients
        self.sos = signal.butter(order, [lowcut, highcut], btype='bandpass', fs=self.fs, output='sos')     

        return signal.sosfiltfilt(self.sos, data, axis=0), self.sos

    def notch(self, data, cutoff = 50, Q = 30):
        """
        Perform Notch filtering. \n
        args: \n
            EEG_data: EEG channel data - Dimension should be (samples, channels) \n
            cutoff: Frequency to remove from the signal \n
            Q: Quality factor \n
                Higher Q → narrower notch (just kills a tiny band).
                Lower Q → wider notch (kills more around the center).
        """
        self.b, self.a = signal.iirnotch(w0=cutoff, Q=Q, fs=self.fs)
        return signal.filtfilt(self.b, self.a, data, axis=0)
    
    def lowpass_filter(self, data, cutoff = 5, order = 4):
        nyquist = 0.5 * self.fs
        normal_cutoff = cutoff / nyquist
        sos = signal.butter(order, normal_cutoff, btype = 'low', output = 'sos')
        return signal.sosfiltfilt(sos, data, axis=0), sos
    
    def visualize_filters(self, EEG_before_filtering, EEG_after_filtering, select_channel = 1, sos = None):
        """
        Figure 1: Visualize Butterworth bandpass filter characterization \n
        Figure 2: Visualize Power Spectral Density - That is dominant frequencies in the data \n
        args: \n
            Data before and after filtering - Dimension should be (samples, channels) \n
            select_channel: Which channel to display - Goes from 1 to max_channel
        """
        sel_ch = select_channel - 1

        # Figure 1
        plt.figure(figsize=(10, 5))
        w, h = signal.sosfreqz(sos, fs=self.fs)
        plt.semilogx(w, 20*np.log10(np.maximum(np.abs(h), 1e-12)))
        plt.title(f'Butterworth BP {self.band_lowcut}-{self.band_highcut} Hz (fs={self.fs})')
        plt.xlabel('Frequency [Hz]'); plt.ylabel('Amplitude [dB]'); 
        plt.grid(True)
        #plt.xlim(-0.1, 100)
        plt.show()

        # Figure 2
        plt.figure(figsize=(10, 5))
        f_original, Pxx_original = signal.welch(EEG_before_filtering[:, sel_ch], fs=self.fs, nperseg=256)
        f_filtered, Pxx_filtered = signal.welch(EEG_after_filtering[:, sel_ch], fs=self.fs, nperseg=256)

        plt.semilogy(f_original, Pxx_original, label='Original Signal')
        plt.semilogy(f_filtered, Pxx_filtered, label='Filtered Signal')
        plt.title("Power Spectral Density Before and After Filtering")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density")
        plt.legend()
        plt.grid()
        plt.show()
    
    def zscore_within_channel(self, data : np.ndarray, mode : str = 'within_ch') -> np.ndarray:
        """
        Standalize data with zscore within or across channels

        Parameters
        ----------
        data : np.ndarray (samples, channels)
            Dataset to standalize
        mode : str
            How to perform zscore on dataset. Example: 'within_ch' or 'across_ch' (default = 'within_ch')
        
        Returns
        -------
        :returns: z-scored data
        """
        
        rts = []
        if mode == 'within_ch':
            mean = np.mean(data, axis=0, keepdims=False)
            std  = np.std(data, axis=0, keepdims=False)
        elif mode == 'across_ch':
            mean = np.mean(data, keepdims=False)
            std  = np.std(data, keepdims=False)
        
        for i in range(data.shape[1]):
            if np.isscalar(mean):
                rts.append( (data[:, i] - mean) / (std + 1e-8) )
            else:
                rts.append( (data[:, i] - mean[i]) / (std[i] + 1e-8) )
        
        return np.array(rts).T
    

class EEG_preprocessing(Filtering):
    def __init__(self, fs = 125):
        super().__init__()
        
        self.channel_names = [
        "Fp1", "Fp2",   # frontal pole
        "C3",  "C4",    # central
        "T5",  "T6",    # temporal (posterior)
        "O1",  "O2",    # occipital
        "F7",  "F8",    # temporal (anterior)
        "F3",  "F4",    # frontal
        "T3",  "T4",    # temporal (mid)
        "P3",  "P4"     # parietal
        ]
        self.fs = fs

    def trim_trial_periods(self,
                           EEG_bandpass : np.ndarray,
                           all_markers: np.ndarray,
                           markers_idx: np.ndarray,
                           extract_event : str = 'ALL') -> tuple[np.ndarray, int]:
        """
        Trims the trial periods from the markers array.

        Parameters
        ----------
        EEG_bandpass : np.ndarray 
            Input of EEG data bandpass filtered
        all_markers : np.ndarray 
            Input of all markers array, including zero values
        markers_idx : np.ndarray
            Indices of non-zero markers. Can be extracted using np.nonzero()[0]
        extract_event : str
            Choise which segment of data to extract (For example: 'ALL', 'CONTRACT', 'RELEASE', 'REST')

        Returns
        ----------
        :return: List of trimmed segments
        :return: Int of the total amount of epochs for one experiment
        """
        if extract_event == 'ALL':
            START_TRIAL_MARKER = 10
            END_TRIAL_MARKER = {10, 201}
            TRIAL_PERIOD = 8            # 8         - 3sec period : 9
        elif extract_event == 'REST':
            START_TRIAL_MARKER = 10
            END_TRIAL_MARKER = {20, 201}
            TRIAL_PERIOD = 2                        # 3
        elif extract_event == 'CONTRACT':
            START_TRIAL_MARKER = 20
            END_TRIAL_MARKER = {30, 201}
            TRIAL_PERIOD = 4            # 3
        elif extract_event == 'RELEASE':
            START_TRIAL_MARKER = 30
            END_TRIAL_MARKER = {10, 201}
            TRIAL_PERIOD = 2            # 3
        else:
            raise ValueError(f'{extract_event} is not valid event type')

        START_TRIM_MARKER = 101
        END_TRIM_MARKER = 102

        enter_trial = False             # Controls when to enter a new trial period
        enter_trim = False              # Controls when to enter a new trim period
        period_extrated = False         # Controls when the boundaries of an entire trial is selected
        num_epochs = 0                  # Increment each time a trial is appended
        st = 0                          # Start boundary index
        ed = 0                          # End boundary index
        data = EEG_bandpass.copy()      # Data
        resampled_data = []             # Resampled data to fit with desired frequency

        # Loop over non-zero indices where markers are present
        for i in range(len(markers_idx)):
            
            # Get the actual index in the markers array plus the next index
            mark_idx = markers_idx[i]
            mark_idx_next = markers_idx[i+1] if i + 1 < len(markers_idx) else None
            
            # Prevent iteration to go out of bounds. RELEASE require END_MARKER to trim the last trial
            if mark_idx_next is None and extract_event != 'RELEASE':
                continue

            # Enter only at the start of the first trim period
            elif all_markers[mark_idx] == START_TRIM_MARKER and not enter_trim:
                st = mark_idx
                enter_trim = True
            
            # Enter only in the end of the first trim period
            elif all_markers[mark_idx] == END_TRIM_MARKER and enter_trim:
                ed = mark_idx
                period_extrated = True
            
            # Enter only at the start of each trial period
            elif all_markers[mark_idx] == START_TRIAL_MARKER and not enter_trial:
                st = mark_idx
                enter_trial = True

            # Enter only at the end of each trial period when the START and END marker is different.
            elif all_markers[mark_idx] in END_TRIAL_MARKER and enter_trial:
                ed = mark_idx - 1
                period_extrated = True
            
            # Enter only in the end of each trial period, if START and END marker belong to the same mark
            elif all_markers[mark_idx_next] in END_TRIAL_MARKER and enter_trial and START_TRIAL_MARKER in END_TRIAL_MARKER:
                ed = mark_idx_next - 1
                period_extrated = True
                
            if period_extrated:
                period_extrated = False
                
                if enter_trim:
                    trim_data = data[st:ed, :]
                    n_samples = trim_data.shape[0]
                    real_fs = n_samples / TRIAL_PERIOD
                    target_len = int(np.round(n_samples * (self.fs / real_fs)))

                    enter_trim = False
                    #print(f'Trial from {st} to {ed}, len: {ed - st}')
                    #print(f'Time period {trim_period}\n Real fs: {real_fs} Hz\n Target len: {target_len}\n Original len: {n_samples}')
                     
                elif enter_trial:
                    trim_data = data[st:ed, :]
                    n_samples = trim_data.shape[0]
                    real_fs = n_samples / TRIAL_PERIOD
                    target_len = int(np.round(n_samples * (self.fs / real_fs)))

                    EEG_resampled = resample(trim_data, target_len, axis=0)
                    resampled_data.append(EEG_resampled)                  
                    
                    enter_trial = False
                    num_epochs += 1             
                    #print(f'Trial from {st} to {ed}, len: {ed - st}')
                    #print(f'Time period {trial_period}\n Real fs: {real_fs} Hz\n Target len: {target_len}\n Original len: {n_samples}')           
        
        return np.concatenate(resampled_data, axis=0), num_epochs
    
    def preprocessing_routine(self,
                              raw_eeg : np.ndarray, 
                              bandpass_lowcut : int = 2, 
                              bandpass_highcut : int = 32, 
                              all_markers : list = None,
                              extract_event : str = 'all',
                              **kwargs) -> tuple[np.ndarray, int]:
        '''
        Performs the full preprocessing routine:
        1) Notch + Bandpass filter
        2) Resample + z-score standardization + Secmentation into epochs

        Parameters
        ----------
        raw_eeg : np.ndarray
            This holds keys for a specfic class (finger). NOTE - If raw_eeg is a list, it will be converted to a dict with key 'single_class'. 2D array - Dim(samples, channels)
        bandpass_lowcut : int
            Lowpass frequency
        bandpass_highcut : int
            Highpass frequency
        all_markers : list
            A log of all marker data including zero
        extract_event : str
            Choise which segment of data to extract (For example: 'ALL', 'CONTRACT', 'RELEASE', 'REST')
        
        Return
        ------
        :return: np.ndarray of normalized EEG data
        :return: Int of the total amount of epochs for one experiment
        '''

        # --------------------------------------------------
        # 1) NOTCH + BANDPASS FILTER
        # --------------------------------------------------


        EEG_filter_ins = Filtering(fs = self.fs)
        
        EEG_notch = EEG_filter_ins.notch(raw_eeg, cutoff=50, Q=30)
        EEG_bandpass, _ = EEG_filter_ins.butter_bandpass(EEG_notch, lowcut=bandpass_lowcut, highcut=bandpass_highcut, order=4)

        # --------------------------------------------------
        # 2) RESAMPLE + Trim
        # --------------------------------------------------
        markers_idx = np.nonzero(all_markers)[0]
        
        EEG_trim, num_epochs = self.trim_trial_periods(EEG_bandpass = EEG_bandpass, 
                                        all_markers = all_markers, 
                                        markers_idx = markers_idx, 
                                        extract_event = str.upper(extract_event))
        
        # --------------------------------------------------
        # 3) Z-SCORE STANDARDIZATION
        # --------------------------------------------------
        EEG_norm = EEG_filter_ins.zscore_within_channel(EEG_trim, mode = 'within_ch')

        return EEG_norm, num_epochs
    
    def reject_channel(self, signal, print_rej_ch=False):
        mean_uV = np.mean(np.abs(signal), axis=0)
        std_uV = np.std(signal, axis=0)

        bad_mean = mean_uV > (np.mean(mean_uV) + 2*np.std(mean_uV))
        bad_std  = std_uV > (np.mean(std_uV) + 2*np.std(std_uV))
        keep_idx = np.where(~(bad_mean | bad_std))[0]
        
        if print_rej_ch:
            reject_idx = np.where((bad_mean | bad_std))[0]
            for i in reject_idx:
                print(f"Channel which is rejected: {self.channel_names[i]}")

        return signal[:, keep_idx], keep_idx

    def preprocessing_routine_OLD(self,
                            raw_eeg, 
                            bandpass_lowcut = 2, 
                            bandpass_highcut = 32,  
                            num_epochs = 35, 
                            trial_period = 9,
                            **kwargs):
        '''
        Performs the full preprocessing routine:
        1) Notch + Bandpass filter
        2) Resample + z-score standardization + Secmentation into epochs

        :param dict raw_eeg: This holds keys for a specfic class (finger). NOTE - If raw_eeg is a list, it will be converted to a dict with key 'single_class'. 2D array - Dim(samples, channels)
        :param int bandpass_lowcut: Lowpass frequency
        :param int bandpass_highcut: Highpass frequency
        :param int num_channels: Number of activated channels
        :param int num_epochs: Number of epochs (trials) in the dataset
        :param int trial_period: Time of each epoch

        :return dict EEG_norm: Dict with normalized continuous EEG data - EEG_norm[keys()](samples, channels)
        :return dict EEG_epoch: Dict with epoched EEG data - EEG_epoch[keys()](epochs, samples_per_epoch, channels)
        :return dict EEG_epoch_mean: Dict with mean over epochs - EEG_epoch_mean[keys()](samples_per_mean_epoch, channels)
        '''
        print('\n-----------------------\n'
              'Old preprocessing routine\n'
              '-------------------------\n')
        # For MRCP:     0.3 - 4 Hz
        # For MU ERP:   6 - 13 Hz
        # For BETA ERP: 14 - 30 Hz

        # --------------------------------------------------
        # 1) NOTCH + BANDPASS FILTER
        # --------------------------------------------------

        EEG_filter_ins = Filtering(fs = self.fs)

        EEG_notch = EEG_filter_ins.notch(raw_eeg, cutoff = 50, Q = 30)
        EEG_bandpass, _ = EEG_filter_ins.butter_bandpass(EEG_notch, lowcut = bandpass_lowcut, highcut = bandpass_highcut, order = 4)

        # --------------------------------------------------
        # 2) RESAMPLE + Z-SCORE STANDARDIZATION + EPOCHING
        # --------------------------------------------------
        # Iterate over each dict
        n_samples, n_channels = EEG_bandpass.shape
        total_time = num_epochs * trial_period                                # total time in seconds.
        real_fs = n_samples / total_time           # Real frequency
        target_len = int( np.round(n_samples  * (self.fs / real_fs)) )        # The correct number of samples for desired frequency
            
        EEG_resampled = resample(EEG_bandpass, target_len, axis=0)

        EEG_norm = EEG_filter_ins.zscore_within_channel(EEG_resampled)
            
        print(f'Total Time {total_time}\n Real fs: {real_fs} Hz\n Target len: {target_len}\n, Original len: {n_samples}')

        return EEG_norm, num_epochs
    
class EMG_preprocessing(Filtering):
    def __init__(self,
                 fs = 2000,
                 bandpass_lowcut : int = 20,
                 bandpass_highcut : int = 450,
                 num_epochs : int = 30,
                 trial_period : int = 8,
                 trim_period : int = 3,):
        super().__init__()

        self.emg_ch_names = [
        'Channel 1 : Palmaris longus',
        'Channel 2 : Flexor digitorum superficialis',
        'Channel 3 : Flexor pollicis longus',
        ]
        self.fs = fs
        self.lowcut = bandpass_lowcut
        self.highcut = bandpass_highcut
        self.num_epochs = num_epochs
        self.trial_period = trial_period
        self.trim_period = trim_period

    def sliding_rms(self, signal, window_size=10, step_size=5):
        rms_vals_all = []
        for ch in range(signal.shape[1]):
            rms_vals = []
            for start in range(0, len(signal) - window_size + 1, step_size):
                window = signal[start : start + window_size, ch]
                rms_vals.append(np.sqrt(np.mean(window**2)))
            rms_vals_all.append(rms_vals)
        return np.array(rms_vals_all).T

    def data_drift(self, EMG, baseline_period = None):
        '''
        Find the minimum data values as a baseline and subtract it from all data across channels
        '''

        if baseline_period is None:
            data = EMG.copy()
        elif isinstance(baseline_period, list):
            if len(baseline_period) != 2:
                raise ValueError('baseline_period must have two elements')
            st, ed = baseline_period[0], baseline_period[1]
            data = EMG[st:ed, :].copy()
        else:
            raise ValueError('baseline_period must be list of two elements or None')
        
        baseline = np.mean(data, axis=0, keepdims=True)
        print(f'baseline value: {baseline}')
        return data - baseline

    def preprocessing_routine(self,
                              raw_emg : np.ndarray,
                            **kwargs) -> np.ndarray:
        '''
        Performs the full preprocessing routine:
        1) Notch + Bandpass filter
        2) Resample + z-score standardization + Secmentation into epochs

        :param dict raw_emg: This holds keys for a specfic class (finger). NOTE - If raw_emg is a list, it will be converted to a dict with key 'single_class'. 2D array - Dim(samples, channels)
        :param int bandpass_lowcut: Lowpass frequency
        :param int bandpass_highcut: Highpass frequency
        :param int num_channels: Number of activated channels
        :param int num_epochs: Number of epochs (trials) in the dataset
        :param int trial_period: Time of each epoch

        :return np.ndarray EMG: Normalized continuous EMG data - EMG(samples, channels)
        '''
        print('\n--------------------------\n'
              'Process to obtain EMG data\n'
              '--------------------------\n')
        # ---------------------------#
        # 1) NOTCH + BANDPASS FILTER #
        # ---------------------------#
        EMG_filter_ins = Filtering(fs = self.fs)        
        
        EMG_notch = EMG_filter_ins.notch(raw_emg, cutoff=50, Q=30)
        EMG_bandpass, _ = EMG_filter_ins.butter_bandpass(EMG_notch, lowcut = self.lowcut, highcut = self.highcut, order=4)

        # --------#
        # 2) TRIM #
        # --------#
        trim_idx_102 = self.fs * self.trim_period
        trim_idx_201 = (self.fs * self.trial_period * self.num_epochs) + trim_idx_102
        EMG_trim = EMG_bandpass[trim_idx_102 : trim_idx_201, :]
        print(f"Trim 102 idx {trim_idx_102}\n"
            f"Trim 201 idx {trim_idx_201}\n"
            f'EMG_trim shape: {EMG_trim.shape}\n')
        
        # ------------#
        # 3) RESAMPLE #
        # ------------#
        total_time = self.num_epochs * self.trial_period                              # total time in seconds.
        n_samples = EMG_trim.shape[0]
        real_fs = n_samples / total_time                              # Real frequency
        target_len = int( np.round(n_samples  * (self.fs / real_fs)) )            # The correct number of samples for desired frequency

        if n_samples == target_len:
            resample_temp = EMG_trim
        else:
            resample_temp = resample(EMG_trim, target_len, axis = 0)
            print(f'Did resample from {n_samples} to {target_len}')

        EMG = EMG_filter_ins.zscore_within_channel(resample_temp, mode = 'within_ch')

        print(f'Total Time {total_time} s\n'
            f'Real fs: {real_fs} Hz\n'
            f'Target len: {target_len}\n'
            f'EMG original len: {n_samples}\n')

        return EMG, self.num_epochs

    def preprocessing_routine_rms(self,
                            raw_emg : np.ndarray,
                            sample_window : int = 200,
                            window_stepsize : int = 50,
                            **kwargs) -> np.ndarray:
        '''
        Performs the full preprocessing routine:
        1) Notch + Bandpass filter
        2) Resample + z-score standardization + Secmentation into epochs

        :param dict raw_emg: This holds keys for a specfic class (finger). NOTE - If raw_emg is a list, it will be converted to a dict with key 'single_class'. 2D array - Dim(samples, channels)
        :param int bandpass_lowcut: Lowpass frequency
        :param int bandpass_highcut: Highpass frequency
        :param int num_channels: Number of activated channels
        :param int num_epochs: Number of epochs (trials) in the dataset
        :param int trial_period: Time of each epoch
        :param int sample_window: Amount of samples in the sliding window.
        :param int window_stepsize: Sliding window step size. This means the step size over the signal.

        :return np.ndarray RMS:  Normalized continuous RMS data - RMS(samples, channels)
        '''
        print('\n--------------------------\n'
              'Process to obtain RMS data\n'
              '--------------------------\n')
        # ---------------------------#
        # 1) NOTCH + BANDPASS FILTER #
        # ---------------------------#
        EMG_filter_ins = Filtering(fs = self.fs)

        EMG_notch = EMG_filter_ins.notch(raw_emg, cutoff=50, Q=30)
        EMG_bandpass, _ = EMG_filter_ins.butter_bandpass(EMG_notch, lowcut = self.lowcut, highcut = self.highcut, order=4)

        # --------#
        # 2) TRIM #
        # --------#
        trim_idx_102 = self.fs * self.trim_period
        trim_idx_201 = (self.fs * self.trial_period * self.num_epochs) + trim_idx_102
        EMG_trim = EMG_bandpass[trim_idx_102 : trim_idx_201, :]
        print(f"Trim 102 idx {trim_idx_102}\n"
           f"Trim 201 idx {trim_idx_201}\n"
           f'EMG_trim shape: {EMG_trim.shape}\n')
        
        # -------#
        # 3) RMS #
        # -------#
        RMS_temp = self.sliding_rms(signal = EMG_trim, window_size = sample_window, step_size = window_stepsize)

        # ------------#
        # 4) RESAMPLE #
        # ------------#
        total_time = self.num_epochs * self.trial_period                              # total time in seconds.
        n_samples = RMS_temp.shape[0]
        real_fs = n_samples / total_time                              # Real frequency
        ceil_fs = int( np.ceil(real_fs) )                                   # Get as close to the real fs
        target_len = int( np.round(n_samples  * (ceil_fs / real_fs)) )            # The correct number of samples for desired frequency


        if n_samples == target_len:
            resample_temp = RMS_temp
        else:
            resample_temp = resample(RMS_temp, target_len, axis = 0)
            print(f'Did resample from {n_samples} to {target_len}')

        RMS = EMG_filter_ins.zscore_within_channel(resample_temp, mode = 'within_ch')

        print(f'\nTotal Time {total_time} s\n'
            f'Real fs: {real_fs} Hz\n'
            f'Target len: {target_len}\n'
            f'EMG original len: {EMG_trim.shape[0]}\n'
            f'RMS len: {RMS.shape[0]}\n')

        return RMS, self.num_epochs
    

if '__main__' == __name__:
    pass