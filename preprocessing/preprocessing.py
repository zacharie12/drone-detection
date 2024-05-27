from scipy.signal import butter, filtfilt, periodogram, medfilt, lfilter
from drone_detection.recording_class.recording import Recording, FrequencyVector
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

def HPS(recording, nfft_factor=1, numProd=6): # harnomic product spectrum
    nfft = recording.audio_data.length * nfft_factor
    fs = recording.sampling_frequency
    f = np.arange(nfft) / nfft
    xf = np.fft.fft(recording.audio_data.values, nfft)
    # Keep magnitude of spectrum at positive frequencies
    xf = np.abs(xf[f < 0.5])
    f = f[f < 0.5]
    N = f.size

    # Downsample-multiply
    smallestLength = int(np.ceil(N / numProd))
    hps = xf[:smallestLength].copy()
    for i in range(2, numProd + 1):
        hps *= xf[::i][:smallestLength]
    f = f[:smallestLength] * fs

    a=5
    return f, hps


