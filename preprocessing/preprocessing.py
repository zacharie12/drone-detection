from scipy.signal import butter, filtfilt, periodogram, medfilt, lfilter
from recording_class.recording import Recording, FrequencyVector
import numpy as np
from scipy.interpolate import interp1d
from scipy.fftpack import next_fast_len

def bpf(recording, low, high, order=6, type='butter'):

    def butter_bandpass(order, low, high, fs):
        nyquist = 0.5 * fs
        low = low / nyquist
        high = high / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(data, order, low, high, fs):
        b, a = butter_bandpass(order, low, high, fs)
        y = lfilter(b, a, data)
        return y

    data = recording.audio_data
    fs = recording.sampling_frequency
    if type == 'butter':
        return Recording(butter_bandpass_filter(data.values, order, low, high, fs), fs, recording.name)
    else:
        return recording

def HPS(recording, nfft_factor=4, numProd=6, hanning_window=True, pre_whiten=True): # harnomic product spectrum
    nfft = recording.audio_data.length * nfft_factor
    fs = recording.sampling_frequency
    f = np.arange(nfft) / nfft
    if hanning_window:
        data = recording.audio_data.values * np.hanning(recording.audio_data.values.size)
        data -= np.mean(data)
    else:
        data = recording.audio_data.values
    xf = np.fft.fft(data, nfft)
    # Keep magnitude of spectrum at positive frequencies
    xf = np.abs(xf[f < 0.5])
    f = f[f < 0.5]
    N = f.size
    freq_resolution = f[1] * fs
    # pre whitening the spectrum
    if pre_whiten:
        MEDIAN_FILTER_WINDOW_SIZE_HZ = 50
        xf_db = np.log(np.abs(xf))
        xf_db_whitened = xf_db - medfilt(xf_db, int(MEDIAN_FILTER_WINDOW_SIZE_HZ/freq_resolution/2)*2+1)
        xf = np.exp(xf_db_whitened)
        xf_db_whitened[xf_db_whitened < 0] = 0
        # xf_db_whitened[xf_db_whitened < 1] = 1
    # Downsample-multiply
    smallestLength = int(np.ceil(N / numProd))
    hps = xf_db_whitened[:smallestLength].copy()
    for i in range(2, numProd + 1):
        hps *= xf_db_whitened[::i][:smallestLength] / np.sum(xf_db_whitened[::i][:smallestLength])
    f = f[:smallestLength] * fs

    a=5
    return f, hps


