import numpy as np
from scipy.signal import spectrogram
import plotly.graph_objs as go
from plotly.subplots import make_subplots


class Vector:
    def __init__(self, arr, sampling_frequency, x=None):
        self.sampling_frequency = sampling_frequency
        self.length = arr.size
        self.y = arr
        self.x = np.arange(arr.size) if x is None else x

    def plot(self, title=[]):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.x, y=self.y))
        fig.update_layout(title_text=title)
        fig.show(renderer="browser")


class TimeVector(Vector):
    def __init__(self, arr, sampling_frequency):
        self.sampling_frequency = sampling_frequency
        self.length = arr.size
        self.values = arr
        self.times = self.create_time()

    def create_time(self):
        return np.arange(self.length) / self.sampling_frequency

    def rms(self):
        return np.sqrt(np.mean(self.values**2))

    def crest_factor(self):
        return np.ptp(self.values) / np.std(self.values)


class FrequencyVector(Vector):
    def __init__(self, data, is_time_vector=True, nfft=None):
        self.sampling_frequency = data.sampling_frequency
        self.original_data_size = data.values.size
        self.nfft = data.values.size if nfft is None else nfft
        self.values = self.calc_psd(data.values) if is_time_vector else data
        self.frequencies = self.create_freqs()
        self.max_freq = self.frequencies[-1]
        self.freq_res = self.frequencies[1]

    def create_freqs(self):
        return np.fft.rfftfreq(self.nfft, d=1/self.sampling_frequency)

    def calc_psd(self, data):
        return np.abs(np.fft.rfft(data) * 2 / self.nfft)**2

    def frequency_band_rms(self, low=0, high=None):
        low = np.max((int(low / self.freq_res), 0))
        high = np.min((int(high / self.freq_res), self.nfft)) if high is not None else self.nfft
        return np.sqrt(np.mean(self.values[low:high]) * self.original_data_size) / 2


class Recording:
    def __init__(self, audio_data, sampling_frequency, name=None):
        self.audio_data = TimeVector(audio_data, sampling_frequency)
        self.psd = FrequencyVector(self.audio_data)
        self.sampling_frequency = sampling_frequency
        self.name = name
        self.cropped_data = None
        self.duration = self.audio_data.length / self.sampling_frequency

    def crop(self, start_sec, end_sec, overwrite=False):
        start_index = int(start_sec * self.sampling_frequency)
        end_index = int(end_sec * self.sampling_frequency)
        if overwrite:
            self.audio_data = TimeVector(self.audio_data.values[start_index:end_index], self.sampling_frequency)
        return TimeVector(self.audio_data.values[start_index:end_index], self.sampling_frequency)

    def create_spectrogram(self, use_cropped=True):
        data = self.cropped_data if use_cropped and self.cropped_data is not None else self.audio_data
        freqs, times, Sxx = spectrogram(data.values, fs=self.sampling_frequency)
        return freqs, times, Sxx

    def play_audio(self, cropped=True):
        import sounddevice as sd
        data = self.cropped_data if cropped and self.cropped_data is not None else self.audio_data
        sd.play(data.values, self.sampling_frequency)
        sd.wait()

    def visualize(self, max_freq=1000):
        fig = make_subplots(rows=3, cols=1, subplot_titles=('Time Domain Signal', 'Power Spectrum', 'Spectrogram'))
        # Plot time domain signal
        fig.add_trace(go.Scatter(x=self.audio_data.times,
                                 y=self.audio_data.values,
                                 mode='lines',
                                 name='Time Domain Signal'), row=1, col=1)
        # add horizontal lines where the cropping is
        # Create horizontal lines for start and end seconds
        # shapes = []
        # if self.start_time is not None:
        #     shapes.append({'type': 'line',
        #                    'x0': self.start_time,
        #                    'y0': min(self.audio_data.values),
        #                    'x1': self.start_time,
        #                    'y1': max(self.audio_data.values),
        #                    'line': {'color': 'red', 'width': 2, 'dash': 'dash'}})
        # if self.end_time is not None:
        #     shapes.append({'type': 'line',
        #                    'x0': self.end_time,
        #                    'y0': min(self.audio_data.values),
        #                    'x1': self.end_time,
        #                    'y1': max(self.audio_data.values),
        #                    'line': {'color': 'red', 'width': 2, 'dash': 'dash'}})
        # fig.update_layout(shapes=shapes)
        # Plot frequency domain (power spectrum)
        psd_freqs = self.psd.frequencies
        psd_values = self.psd.values
        if max_freq is None:
            max_freq = psd_freqs[-1]
        freq_mask = psd_freqs <= max_freq
        fig.add_trace(go.Scatter(x=psd_freqs[freq_mask],
                                 y=psd_values[freq_mask],
                                 mode='lines',
                                 name='Power Spectrum',
                                 yaxis='y2'), row=2, col=1)

        # Plot spectrogram
        freqs, times, Sxx = self.create_spectrogram()
        freq_mask = freqs <= max_freq
        Sxx_db = 10 * np.log10(Sxx)
        fig.add_trace(go.Heatmap(z=Sxx_db,
                                 x=times,
                                 y=freqs[freq_mask],
                                 colorscale='Viridis',
                                 zmin=np.min(Sxx_db),
                                 zmax=np.max(Sxx_db),
                                 name='Spectrogram'), row=3, col=1)

        # Update layout & show plot
        fig.update_layout(title=self.name)
        # Update x-axis titles
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        # Update y-axis titles
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="Power", row=2, col=1)
        fig.update_yaxes(title_text="Frequency (Hz)", row=3, col=1)
        fig.show(renderer='browser')



