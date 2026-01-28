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
    
    def zscore_within_channel(self, data):
        """
        data: np.ndarray of shape (samples, channels)
        returns z-scored EEG per channel
        """
        
        rts = []
        # Within channels
        mean = np.mean(data, axis=0, keepdims=False)
        std  = np.std(data, axis=0, keepdims=False)

        # Across channels
        #mean = np.mean(data, keepdims=False)
        #std  = np.std(data, keepdims=False)
        
        if len(data.shape) == 1:
            rts.append( (data - mean) / (std + 1e-8) )
            return np.array(rts).T

        for i in range(data.shape[1]):
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
    
    def preprocessing_routine(self,
                            raw_eeg, 
                            bandpass_lowcut = 2, 
                            bandpass_highcut = 32, 
                            num_channels = 16, 
                            num_epochs = 30, 
                            trial_period = 9):
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

        if isinstance(raw_eeg, dict) is False:
            EEG_raw = {}
            key_name = 'single_class'
            EEG_raw[key_name] = raw_eeg
        else:
            EEG_raw = raw_eeg.copy()

        # For MRCP:     0.3 - 4 Hz
        # For MU ERP:   6 - 13 Hz
        # For BETA ERP: 14 - 30 Hz

        # --------------------------------------------------
        # 1) NOTCH + BANDPASS FILTER
        # --------------------------------------------------

        EEG_filter_ins = Filtering(fs = self.fs)

        EEG_notch = {}
        EEG_bandpass = {}
        sos = {}

        for f in EEG_raw.keys():
            EEG_notch[f] = EEG_filter_ins.notch(EEG_raw[f], cutoff=50, Q=30)
            EEG_bandpass[f], sos[f] = EEG_filter_ins.butter_bandpass(EEG_notch[f], lowcut=bandpass_lowcut, highcut=bandpass_highcut, order=4)

        # --------------------------------------------------
        # 2) RESAMPLE + Z-SCORE STANDARDIZATION + EPOCHING
        # --------------------------------------------------
        EEG_resampled = {}
        EEG_norm = {}
        EEG_epoch = {}
        EEG_epoch_mean = {}

        # Iterate over each dict
        for finger_movement in EEG_raw.keys():
            total_time = num_epochs * trial_period                                # total time in seconds.
            real_fs = len(EEG_bandpass[finger_movement]) / (total_time)           # Real frequency
            target_len = int( len(EEG_bandpass[finger_movement])  * (self.fs / real_fs) )        # The correct number of samples for desired frequency
            

            EEG_resampled[finger_movement] = resample(EEG_bandpass[finger_movement], target_len, axis=0)
            
            print(f'For finger motion: {finger_movement}')
            print(f'Total Time {total_time}\n Real fs: {real_fs} Hz\n Target len: {target_len}\n Original len: {len(EEG_bandpass[finger_movement])}')

            EEG_norm[finger_movement] = EEG_filter_ins.zscore_within_channel(EEG_resampled[finger_movement])
            
            # EPOCHS
            EEG_samples_per_epoch = EEG_norm[finger_movement].shape[0] // num_epochs      

            EEG_epoch[finger_movement] = EEG_norm[finger_movement].reshape(num_epochs, EEG_samples_per_epoch, num_channels)

            EEG_epoch_mean[finger_movement] = EEG_epoch[finger_movement].mean(axis=0)

            print(f'EEG resample shape: {EEG_resampled[finger_movement].shape}\n'
                f'EEG norm shape: {EEG_norm[finger_movement].shape}\n'
                f'EEG epoch shape: {EEG_epoch[finger_movement].shape}\n'
                f'EEG mean epoch shape: {EEG_epoch_mean[finger_movement].shape}\n')
        
        return EEG_norm, EEG_epoch, EEG_epoch_mean

    
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

class EMG_preprocessing(Filtering):
    def __init__(self, fs = 2000):
        super().__init__()

        EMG_CH_NAMES = [
        'Channel 1 : Palmaris longus',
        'Channel 2 : Flexor digitorum superficialis',
        'Channel 3 : Flexor pollicis longus',
        ]
        self.fs = fs

    def zscore_within_channel(self, eeg):
        """
        eeg: np.ndarray of shape (samples, channels)
        returns z-scored EEG per channel
        """
        
        eeg_return = []
        mean = np.mean(eeg, axis=0, keepdims=False)
        std  = np.std(eeg, axis=0, keepdims=False)

            
        if len(eeg.shape) == 1:
            eeg_return.append( (eeg[:] - mean) / (std + 1e-8) )
            return np.array(eeg_return).T

        for i in range(eeg.shape[1]):
            eeg_return.append( (eeg[:, i] - mean[i]) / (std[i] + 1e-8) )
        
        return np.array(eeg_return).T

    def sliding_rms(self, signal, window_size=10, step_size=5):
        rms_vals_all = []
        for ch in range(signal.shape[1]):
            rms_vals = []
            for start in range(0, len(signal) - window_size + 1, step_size):
                window = signal[start : start + window_size, ch]
                rms_vals.append(np.sqrt(np.mean(window**2)))
            rms_vals_all.append(rms_vals)
        return np.array(rms_vals_all).T

    def preprocessing_routine_rms(self,
                                raw_emg,
                                bandpass_lowcut = 20,
                                bandpass_highcut = 450,
                                num_channels = 3,
                                num_epochs = 30,
                                trial_period = 8,
                                sample_window = 200,
                                window_stepsize = 50):
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

        :return dict RMS: Dict with normalized continuous RMS data - RMS[keys()](samples, channels)
        :return dict RMS_epoch: Dict with epoched RMS data - RMS_epoch[keys()](epochs, samples_per_epoch, channels)
        :return dict RMS_epoch_mean: Dict with mean over epochs - RMS_epoch_mean[keys()](samples_per_mean_epoch, channels)
        '''
        if isinstance(raw_emg, dict) is False:
            EMG_raw = {}
            key_name = 'single_class'
            EMG_raw[key_name] = raw_emg.copy()
        else:
            EMG_raw = raw_emg.copy()
        # --------------------------------------------------
        # 1) NOTCH + BANDPASS FILTER
        # --------------------------------------------------

        EMG_filter_ins = Filtering(fs = self.fs)

        EMG_notch = {}
        EMG_bandpass = {}
        sos = {}

        for f in EMG_raw.keys():
            EMG_notch[f] = EMG_filter_ins.notch(EMG_raw[f], cutoff=50, Q=30)
            EMG_bandpass[f], sos[f] = EMG_filter_ins.butter_bandpass(EMG_notch[f], lowcut=bandpass_lowcut, highcut=bandpass_highcut, order=4)


        # --------------------------------------------------
        # 2) RESAMPLE + EPOCH
        # --------------------------------------------------
        rms_temp = {}
        RMS = {}
        RMS_epoch = {}
        RMS_epoch_mean = {}

        for f_key in EMG_raw.keys():
            # Sliding RMS
            rms_temp[f_key] = self.sliding_rms(signal = EMG_bandpass[f_key], window_size = sample_window, step_size = window_stepsize)

            # Resample + normalization
            total_time = num_epochs * trial_period                              # total time in seconds.
            real_fs = len(rms_temp[f_key]) / (total_time)                              # Real frequency
            ceil_fs = int( np.ceil(real_fs) )                                   # Get as close to the real fs
            target_len = int( len(rms_temp[f_key])  * (ceil_fs / real_fs) )            # The correct number of samples for desired frequency

            resample_temp = resample(rms_temp[f_key], target_len, axis = 0)
            RMS[f_key] = self.zscore_within_channel(resample_temp)

            # EPOCHS
            RMS_samples_per_epoch = RMS[f_key].shape[0] // num_epochs
            RMS_epoch[f_key] = RMS[f_key].reshape(num_epochs, RMS_samples_per_epoch, num_channels)
            RMS_epoch_mean[f_key] = RMS_epoch[f_key].mean(axis = 0)

            print(f'For finger motion: {f_key}')
            print(f'Total Time {total_time} s\n'
                f'Real fs: {real_fs} Hz\n'
                f'Ceil fs: {ceil_fs} Hz\n'
                f'Target len: {target_len}\n'
                f'RMS len before resample: {rms_temp[f_key].shape}\n'
                f'EMG original len: {len(EMG_bandpass[f_key])}\n')
            print(f'RMS resample shape: {RMS[f_key].shape}\n'
                f'RMS epoch shape: {RMS_epoch[f_key].shape}\n'
                f'RMS mean epoch shape: {RMS_epoch_mean[f_key].shape}\n')

        return RMS, RMS_epoch, RMS_epoch_mean

    def preprocessing_routine(self,
                              raw_emg,
                              bandpass_lowcut = 20,
                              bandpass_highcut = 450,
                              num_channels = 3,
                              num_epochs = 30,
                              trial_period = 8):
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

        :return dict EMG: Dict with normalized continuous EMG data - EMG[keys()](samples, channels)
        :return dict EMG_epoch: Dict with epoched EMG data - EMG_epoch[keys()](epochs, samples_per_epoch, channels)
        :return dict EMG_epoch_mean: Dict with mean over epochs - EMG_epoch_mean[keys()](samples_per_mean_epoch, channels)
        '''
        if isinstance(raw_emg, dict) is False:
            EMG_raw = {}
            key_name = 'single_class'
            EMG_raw[key_name] = raw_emg.copy()
        else:
            EMG_raw = raw_emg.copy()
        # --------------------------------------------------
        # 1) NOTCH + BANDPASS FILTER
        # --------------------------------------------------

        EMG_filter_ins = Filtering(fs = self.fs)

        EMG_notch = {}
        EMG_bandpass = {}
        sos = {}
        
        # 2000 * 30 * 8 = 480000
        # Trim 6 seconds = 12000 samples
        # 468000 samples remain 

        for f in EMG_raw.keys():
            EMG_notch[f] = EMG_filter_ins.notch(EMG_raw[f], cutoff=50, Q=30)
            EMG_bandpass[f], sos[f] = EMG_filter_ins.butter_bandpass(EMG_notch[f], lowcut=bandpass_lowcut, highcut=bandpass_highcut, order=4)

        # --------------------------------------------------
        # 2) RESAMPLE + EPOCH
        # --------------------------------------------------
        EMG = {}
        EMG_epoch = {}
        EMG_epoch_mean = {}

        for f_key in EMG_raw.keys():
            # Resample + normalization
            total_time = num_epochs * trial_period                              # total time in seconds.
            real_fs = len(EMG_bandpass[f_key]) / (total_time)                              # Real frequency
            target_len = int( len(EMG_bandpass[f_key])  * (self.fs / real_fs) )            # The correct number of samples for desired frequency

            resample_temp = resample(EMG_bandpass[f_key], target_len, axis = 0)
            EMG[f_key] = self.zscore_within_channel(resample_temp)

            # EPOCHS
            RMS_samples_per_epoch = EMG[f_key].shape[0] // num_epochs
            EMG_epoch[f_key] = EMG[f_key].reshape(num_epochs, RMS_samples_per_epoch, num_channels)
            EMG_epoch_mean[f_key] = EMG_epoch[f_key].mean(axis = 0)

            print(f'For finger motion: {f_key}')
            print(f'Total Time {total_time} s\n'
                f'Real fs: {real_fs} Hz\n'
                f'Target len: {target_len}\n'
                f'EMG original len: {len(EMG_bandpass[f_key])}\n')
            print(f'EMG resample shape: {EMG[f_key].shape}\n'
                f'EMG epoch shape: {EMG_epoch[f_key].shape}\n'
                f'EMG mean epoch shape: {EMG_epoch_mean[f_key].shape}\n')

        return EMG, EMG_epoch, EMG_epoch_mean
    
def main():
    EEG_instance = EEG_preprocessing(fs = 125)
    

if '__main__' == __name__:
    main()