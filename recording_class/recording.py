import numpy as np
from scipy.signal import spectrogram
import plotly.graph_objs as go
from plotly.subplots import make_subplots

class TimeVector:
    def __init__(self, arr, sampling_frequency):
        self.sampling_frequency = sampling_frequency
        self.length = arr.size
        self.values = arr
        self.times = self.create()

    def create(self):
        return np.arange(self.length) / self.sampling_frequency

class FrequencyVector:
    def __init__(self, time_vector, nfft=None):
        self.sampling_frequency = time_vector.sampling_frequency
        self.nfft = time_vector.values.size if nfft is None else nfft
        self.values = self.calc_psd(time_vector.values)
        self.frequencies = self.create_freqs()

    def create_freqs(self):
        return np.fft.rfftfreq(self.nfft, d=1/self.sampling_frequency)

    def calc_psd(self, data):
        return np.abs(np.fft.rfft(data) * 2 / self.nfft)**2

class Recording:
    def __init__(self, audio_data, sampling_frequency, name):
        self.audio_data = TimeVector(audio_data, sampling_frequency)
        self.psd = FrequencyVector(self.audio_data)
        self.sampling_frequency = sampling_frequency
        self.name = name
        self.cropped_data = None

    def crop(self, start_sec, end_sec):
        start_index = int(start_sec * self.sampling_frequency)
        end_index = int(end_sec * self.sampling_frequency)
        self.start_time = start_sec
        self.end_time = end_sec
        self.cropped_data = TimeVector(self.audio_data.values[start_index:end_index], self.sampling_frequency)
        self.psd_cropped = FrequencyVector(self.cropped_data)

    def create_spectrogram(self, use_cropped=True):
        data = self.cropped_data if use_cropped and self.cropped_data is not None else self.audio_data
        freqs, times, Sxx = spectrogram(data.values, fs=self.sampling_frequency)
        return freqs, times, Sxx

    def play_audio(self, cropped=True):
        import sounddevice as sd
        data = self.cropped_data if cropped and self.cropped_data is not None else self.audio_data
        sd.play(data.values, self.sampling_frequency)
        sd.wait()

    def visualize(self):
        fig = make_subplots(rows=3, cols=1, subplot_titles=('Time Domain Signal', 'Power Spectrum', 'Spectrogram'))
        # Plot time domain signal
        fig.add_trace(go.Scatter(x=self.audio_data.times,
                                 y=self.audio_data.values,
                                 mode='lines',
                                 name='Time Domain Signal'), row=1, col=1)
        # add horizontal lines where the cropping is
        # Create horizontal lines for start and end seconds
        shapes = []
        if self.start_time is not None:
            shapes.append({'type': 'line',
                           'x0': self.start_time,
                           'y0': min(self.audio_data.values),
                           'x1': self.start_time,
                           'y1': max(self.audio_data.values),
                           'line': {'color': 'red', 'width': 2, 'dash': 'dash'}})
        if self.end_time is not None:
            shapes.append({'type': 'line',
                           'x0': self.end_time,
                           'y0': min(self.audio_data.values),
                           'x1': self.end_time,
                           'y1': max(self.audio_data.values),
                           'line': {'color': 'red', 'width': 2, 'dash': 'dash'}})
        fig.update_layout(shapes=shapes)
        # Plot frequency domain (power spectrum)
        psd_freqs = self.psd_cropped.frequencies if self.cropped_data is not None else self.psd.frequencies
        psd_values = self.psd_cropped.values if self.cropped_data is not None else self.psd.values
        fig.add_trace(go.Scatter(x=psd_freqs,
                                 y=psd_values,
                                 mode='lines',
                                 name='Power Spectrum',
                                 yaxis='y2'), row=2, col=1)

        # Plot spectrogram
        freqs, times, Sxx = self.create_spectrogram()
        Sxx_db = 10 * np.log10(Sxx)
        fig.add_trace(go.Heatmap(z=Sxx_db,
                                 x=times,
                                 y=freqs,
                                 colorscale='Viridis',
                                 zmin=np.min(Sxx_db),
                                 zmax=np.max(Sxx_db),
                                 name='Spectrogram'), row=3, col=1)

        # Update layout & show plot
        fig.update_layout(title=self.name)
        fig.show(renderer='browser')

