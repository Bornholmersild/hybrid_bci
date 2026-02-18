import numpy as np
import scipy.stats as stats
from scipy.signal import welch
import antropy as ant

class FeatureExtraction():
    def __init__(self, fs = 125):
        self.fs = fs

    def extract_statistical_features(self, signal):
        mean = np.mean(signal)
        median = np.median(signal)
        std_dev = np.std(signal)
        variance = np.var(signal)
        skewness = stats.skew(signal)
        kurtosis = stats.kurtosis(signal)
        range_val = np.ptp(signal)
        
        features = {
            'Mean': mean,
            'Median': median,
            'Standard Deviation': std_dev,
            'Variance': variance,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Range': range_val
        }
    
        return features
    
    def extract_time_domain_features(self, signal):
        rms = np.sqrt(np.mean(signal**2))
        zero_crossings = ((signal[:-1] * signal[1:]) < 0).sum()
        autocorrelation = np.correlate(signal, signal, mode='full')[len(signal)-1]
        mean_abs_dev = np.mean(np.abs(signal - np.mean(signal)))
        max_val = np.max(signal)
        min_val = np.min(signal)
        signal_energy = np.sum(signal**2)
        
        features = {
            'RMS': rms,
            'Zero Crossings': zero_crossings,
            'Autocorrelation': autocorrelation,
            'Mean Absolute Deviation': mean_abs_dev,
            'Max Value': max_val,
            'Min Value': min_val,
            'Signal Energy': signal_energy
        }
        
        return features
    
    def extract_frequency_domain_features(self, signal):
        freqs, psd = welch(signal, self.fs)
        dominant_freq = freqs[np.argmax(psd)]
        total_power = np.sum(psd)
        band_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 40)])
        mean_freq = np.mean(freqs)
        median_freq = np.median(freqs)
        peak_freq = freqs[np.argmax(psd)]
        freq_variance = np.var(freqs)
        
        features = {
            'Dominant Frequency': dominant_freq,
            'Total Power': total_power,
            'Band Power (0.5-40 Hz)': band_power,
            'Mean Frequency': mean_freq,
            'Median Frequency': median_freq,
            'Peak Frequency': peak_freq,
            'Frequency Variance': freq_variance
        }
        
        return features
    
    def extract_entropy_features(self, signal):
        sample_entropy = ant.sample_entropy(signal)
        spectral_entropy = ant.spectral_entropy(signal, sf = self.fs, method='welch')
        perm_entropy = ant.perm_entropy(signal, normalize=True)
        svd_entropy = ant.svd_entropy(signal, order=3, delay=1)
        app_entropy = ant.app_entropy(signal)
        lziv_complexity = ant.lziv_complexity(signal)
        
        features = {
            'Sample Entropy': sample_entropy,
            'Spectral Entropy': spectral_entropy,
            'Permutation Entropy': perm_entropy,
            'SVD Entropy': svd_entropy,
            'Approximate Entropy': app_entropy,
            'LZiv Complexity': lziv_complexity
        }
        
        return features