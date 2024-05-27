from drone_detection.preprocessing.preprocessing import bpf, HPS
from drone_detection.recording_class.recording import Recording, Vector
from plotly.subplots import make_subplots
import plotly.graph_objects as go
class Epoch:

    def __init__(self, recording, name):
        self.recording = recording
        self.name = name

    def visualize(self):
        self.recording.visualize()

    def crop(self, start_sec, end_sec):
        cropped_time_vector = self.bandpassed_recording.crop(start_sec, end_sec)
        self.cropped_recording = Recording(cropped_time_vector.values, self.recording.sampling_frequency, self.name + '_cropped')

    def visualize_cropped(self):
        self.cropped_recording.visualize()

    def play_audio(self):
        self.recording.play_audio()

    def preprocess(self):
        self.bandpassed_recording = bpf(self.recording, 80, 1500)

    def calc_hps(self, use_cropped=False):
         f, hps = HPS(self.cropped_recording) if use_cropped and hasattr(self, 'cropped_recording') else HPS(self.bandpassed_recording)
         self.hps = Vector(hps, self.bandpassed_recording.sampling_frequency, f)

    def window_recording(self, window_length_sec=0.5):
        recording_windows = []
        num_windows = int(self.bandpassed_recording.audio_data.length / (window_length_sec * self.bandpassed_recording.sampling_frequency))
        for i in range(num_windows):
            start = i * window_length_sec
            end = (i + 1) * window_length_sec
            recording_window = self.bandpassed_recording.crop(start, end)
            recording_windows.append(Recording(recording_window.values, self.recording.sampling_frequency, self.name + '_window_' + str(i)))
        self.recording_windows = recording_windows

    def windows_hps(self):
        windows_hps = []
        for i, window in enumerate(self.recording_windows):
            f, hps = HPS(window, 10)
            windows_hps.append(Vector(hps, window.sampling_frequency, f))
        self.windows_hps = windows_hps

    def visualize_windows(self, max_freq=1000):
        i = 0 # Window index
        for w, hps_w in zip(self.recording_windows, self.windows_hps):
            i += 1
            if i > 30:
                break
            fig = make_subplots(rows=3, cols=1, subplot_titles=('Time Domain Signal', 'Power Spectrum', 'HPS'))
            # Plot time domain signal
            fig.add_trace(go.Scatter(x=w.audio_data.times,
                                     y=w.audio_data.values,
                                     mode='lines',
                                     name='Time Domain Signal'), row=1, col=1)
            psd_freqs = w.psd.frequencies
            psd_values = w.psd.values
            if max_freq is None:
                max_freq = psd_freqs[-1]
            freq_mask = psd_freqs <= max_freq
            fig.add_trace(go.Scatter(x=psd_freqs[freq_mask],
                                     y=psd_values[freq_mask],
                                     mode='lines',
                                     name='Power Spectrum',
                                     yaxis='y2'), row=2, col=1)
            # Plot HPS
            fig.add_trace(go.Scatter(x=hps_w.x,
                                     y=hps_w.y,
                                     mode='lines',
                                     name='HPS',
                                     yaxis='y3'), row=3, col=1)
            # Update layout & show plot
            fig.update_layout(title=w.name)
            # Update x-axis titles
            fig.update_xaxes(title_text="Time (s)", row=1, col=1)
            fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
            fig.update_xaxes(title_text="Frequency (s)", row=3, col=1)
            # Update y-axis titles
            fig.update_yaxes(title_text="Amplitude", row=1, col=1)
            fig.update_yaxes(title_text="Power", row=2, col=1)
            fig.update_yaxes(title_text="Power", row=3, col=1)
            fig.show(renderer='browser')







