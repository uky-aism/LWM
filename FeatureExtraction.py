import numpy as np
from scipy.stats import skew, kurtosis
import pywt

class FeatureExtraction:
    
    def __init__(self, data, cal_points, overlap=0):
        """
        Initializes the FeatureExtraction object.
        
        Parameters:
        data (np.array): The time series data.
        cal_points (int): The number of points to calculate.
        overlap (int): The number of overlapping points. Default is 0.
        """
        self.data = data
        self.cal_points = cal_points
        self.overlap = overlap
        
    def sig_cal(self):
        """
        Calculate time domain features.

        Returns:
        np.array: A 2D array with time domain features for each segment.
        """
        step = self.cal_points - self.overlap
        feature_list = ['avg', 'RMS', 'std', 'skew', 'kurtosis', 'crest', 'latitude', 'shapeF', 'impulse']
        chunk_num = int((self.data.shape[0] - self.overlap) / step)
        sig_0 = np.zeros((chunk_num, len(feature_list)))
        
        for i in range(chunk_num):
            sig_calculate1 = self.data[i*step:(i*step+self.cal_points)]
            sig_0[i, 0] = np.mean(sig_calculate1)  # avg
            sig_0[i, 1] = np.sqrt(np.mean(sig_calculate1**2))  # RMS
            smr = (np.mean(np.sqrt(np.abs(sig_calculate1))))**2
            sig_0[i, 2] = np.std(sig_calculate1)  # SD
            sig_0[i, 3] = skew(sig_calculate1)  # skew
            sig_0[i, 4] = kurtosis(sig_calculate1)  # kurtosis
            sig_0[i, 5] = np.max(sig_calculate1) / sig_0[i, 1]  # crest
            sig_0[i, 6] = np.max(np.abs(sig_calculate1)) / smr  # latitude
            sig_0[i, 7] = sig_0[i, 1] / np.mean(np.abs(sig_calculate1))  # shape factor
            sig_0[i, 8] = np.max(np.abs(sig_calculate1)) / np.mean(np.abs(sig_calculate1))  # impulse factor 
        
        return sig_0
    
    def fre_cal(self, fs=50000, nperseg=256*4, noverlap=128*4):
        """
        Calculate frequency domain features.

        Parameters:
        fs (int): Sampling frequency. Default is 50000.
        nperseg (int): Number of points per segment for PSD calculation. Default is 256*5.
        noverlap (int): Number of overlapping points for PSD calculation. Default is 128*5.

        Returns:
        np.array: A 2D array with frequency domain features for each segment.
        """
        step1 = self.cal_points - self.overlap
        chunk_num = int((self.data.shape[0] - self.overlap) / step1)
        feature_list = ['mean_freq', 'aver_freq', 'stab_factor', 'coef_var', 'Fre_skew', 'Fre_kurt', 'RMSratio']
        sig_1 = np.ones((chunk_num, len(feature_list)))
        
        for i in range(sig_1.shape[0]):
            sig_calculate1 = self.data[i*step1:(i*step1+self.cal_points)]
            win = np.ones(nperseg)
            step = nperseg - noverlap
            num_segments = (len(sig_calculate1) - noverlap) // step
            psds = []

            for j in range(num_segments):
                segment = sig_calculate1[j*step : j*step + nperseg]
                win_segment = segment * win
                segment_fft = np.fft.fft(win_segment, n=nperseg)
                segment_fft = segment_fft[:noverlap]
                segment_psd = (np.abs(segment_fft) ** 2) / (fs * noverlap)
                psds.append(segment_psd)
        
            psd = np.mean(psds, axis=0)
            frequencies = np.fft.fftfreq(nperseg, 1/fs)[:(nperseg // 2)]
            f_mean = np.sum(frequencies * psd) / np.sum(psd)
            sigma = np.sqrt(np.sum((frequencies - f_mean)**2 * psd) / len(frequencies))
            K = len(frequencies)
            mean_freq = np.sqrt(np.sum((frequencies**2) * psd) / np.sum(psd))
            aver_freq = np.sqrt(np.sum((frequencies**4) * psd) / np.sum((frequencies**2) * psd))
            stab_factor = np.sum(frequencies**2 * psd) / np.sqrt(np.sum(psd) * np.sum(frequencies**4 * psd))
            coef_var = sigma / f_mean
            Fre_skew = np.sum(((frequencies - f_mean)**3) * psd) / ((sigma**3) * K)
            Fre_kurt = np.sum(((frequencies - f_mean)**4) * psd) / ((sigma**4) * K)
            RMSratio = np.sum(np.sqrt(np.abs(frequencies - f_mean)) * psd) / (np.sqrt(sigma) * K)

            sig_1[i, 0] = mean_freq
            sig_1[i, 1] = aver_freq
            sig_1[i, 2] = stab_factor
            sig_1[i, 3] = coef_var
            sig_1[i, 4] = Fre_skew
            sig_1[i, 5] = Fre_kurt
            sig_1[i, 6] = RMSratio
        
        return sig_1

    def wave_cal(self):
        """
        Calculate wavelet domain features.

        Returns:
        np.array: A 2D array with wavelet domain features for each segment.
        """
        step = self.cal_points - self.overlap
        chunk_num = int((self.data.shape[0] - self.overlap) / step)
        sig_0 = np.zeros((chunk_num, 7))
        
        for i in range(chunk_num):
            sig_calculate1 = self.data[i*step:(i*step+self.cal_points)]
            coeffs = pywt.wavedec(sig_calculate1, 'db1', level=4)
            mean = [np.mean(c) for c in coeffs]
            standard_d = [np.std(c) for c in coeffs]
            sig_0[i, 0] = mean[0]
            sig_0[i, 1] = mean[1]
            sig_0[i, 2] = mean[4]
            sig_0[i, 3] = standard_d[0]
            sig_0[i, 4] = standard_d[1]
            sig_0[i, 5] = standard_d[4]
            sig_0[i, 6] = self.shannon_entropy(coeffs)
        
        return sig_0

    @staticmethod
    def shannon_entropy(coeffs):
        """
        Calculate Shannon entropy of wavelet coefficients.

        Parameters:
        coeffs (list): List of wavelet coefficients.

        Returns:
        float: Shannon entropy value.
        """
        all_coeffs = np.concatenate([c.flatten() for c in coeffs])
        energy = all_coeffs**2
        total_energy = np.sum(energy)
        p = energy / total_energy
        p = p[p > 0]
        entropy = -np.sum(p * np.log2(p))
        return entropy
