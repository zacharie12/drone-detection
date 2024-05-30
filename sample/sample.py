from preprocessing.preprocessing import bpf, HPS
from recording_class.recording import Recording, Vector
from plotly.subplots import make_subplots
from feature_extraction.feature_extraction_sample import FeatureExtractionSample
import plotly.graph_objects as go


class Sample:

    def __init__(self, recording, name=None, label=None, create_label=False):
        # labels: 0 = no drone, 1 = ababil, 2 = hermes, 3 = both
        self.recording = recording
        self.name = name
        self.label = self.set_label(label) if label is not None and create_label else None
        self.preprocess()
        self.calc_hps()
        self.features = FeatureExtractionSample(self).extract_features()

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

    def add_features(self, features):
        if isinstance(features, dict):
            for feat_name, feat_value in features:
                self.features[feat_name] = feat_value

    def set_label(self, label=None):
        if label in [0, 1, 2, 3]:
            self.label = label
        else:
            self.recording.play_audio()
            while True:
                label = input("Enter label (0, 1, 2, 3): ")
                if label in ['0', '1', '2', '3']:
                    self.label = int(label)
                    break
                else:
                    print("Invalid label. Please enter 0, 1, 2 or 3.")

    def predict(self):
        pass

    def remove_recording(self):
        del self.recording
        del self.bandpassed_recording
        if hasattr(self, 'cropped_recording'):
            del self.cropped_recording
        del self.hps










