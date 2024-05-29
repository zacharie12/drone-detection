from drone_detection.epoch.epoch_class import Epoch
from drone_detection.recording_class.recording import Recording
from drone_detection.feature_extraction.feature_extraction_epoch import FeatureExtractionEpoch


class Stream():
    def __init__(self, name=None):
        self.epochs_list = []
        self.epoch_features_list = []
        self.labels = []
        self.name = name

    def set_labels(self, labels):
        self.labels = labels
        if len(self.labels) != len(self.epochs_list):
            raise Warning("Number of labels and epochs do not match")

    def split_long_recording_into_epochs(self, long_recording, window_length_seconds=0.5, limit_num_windows=False):
        def _window_recording(recording, window_length_sec=0.5):
            recording_windows = []
            num_windows = int(recording.audio_data.length / (window_length_sec * recording.sampling_frequency))
            for i in range(num_windows):
                start = i * window_length_sec
                end = (i + 1) * window_length_sec
                recording_window = recording.crop(start, end)
                recording_windows.append(Recording(recording_window.values, recording.sampling_frequency,
                                                   recording.name + '_window_' + str(i)))
            return recording_windows

        epochs = []
        num_windows = int(long_recording.audio_data.length / (window_length_seconds * long_recording.sampling_frequency))
        for i, w in enumerate(_window_recording(long_recording, window_length_seconds)):
            if limit_num_windows and i > limit_num_windows:
                print(f"reached limit of {limit_num_windows} windows")
                break
            print(f"creating epoch of window {i} out of {num_windows}")
            epoch = Epoch(w, f"{long_recording.name}_window_{i}")
            epoch.preprocess()
            epoch.calc_hps()
            epochs.append(epoch)
        self.epochs_list.extend(epochs)

    def add_epochs(self, epochs):
        self.epochs_list.extend(epochs)

    def extract_features_all_epochs(self):
        for i, epoch in enumerate(self.epochs_list):
            if i > 1:
                previous_epochs = self.epochs_list[i-2:i]
            else:
                previous_epochs = None
            if i < len(self.epoch_features_list):
                continue
            self.epoch_features_list.append(self.extract_features_single_epoch(epoch, previous_epochs))

    def extract_features_single_epoch(self, epoch, previous_epochs):
        epoch_features_class = FeatureExtractionEpoch(epoch, previous_epochs)
        epoch_features_class.extract_features()
        return epoch_features_class.get_features()

    def predict_epochs(self): # placeholder
        return None
