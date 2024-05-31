from drone_detection.sample.sample import Sample
from drone_detection.recording_class.recording import Recording
from drone_detection.feature_extraction.feature_extraction_sample import FeatureExtractionSample


class Stream():
    def __init__(self, recording, sample_window_duration_sec=0.5, window_ovelap_sec=0, limit_num_windows=False, num_lookback_samples=3, name=None, labels=None):
        if window_ovelap_sec > sample_window_duration_sec:
            raise ValueError("Overlap duration cannot be greater than sample window duration")
        self.recording = recording
        self.name = name
        self.samples_list = []
        self.split_long_recording_into_samples(
            long_recording=recording,
            window_overlap_seconds=window_ovelap_sec,
            window_length_seconds=sample_window_duration_sec,
            limit_num_windows=limit_num_windows,
            labels=labels)
        self.add_previous_samples_features(num_lookback_samples=num_lookback_samples)

    def manually_label_sample(self, sample_num):
        self.samples_list[sample_num].set_label()

    def manually_label_samples(self, start_sample=0, end_sample=None):
        if end_sample is None:
            end_sample = len(self.samples_list)
        for i in range(start_sample, end_sample):
            self.manually_label_sample(i)

    def split_long_recording_into_samples(self, long_recording, window_length_seconds, window_overlap_seconds, limit_num_windows, labels):
        def _calc_num_windows(long_recording, window_length_seconds, window_overlap_seconds):
            N = long_recording.audio_data.length
            window_size_samples = int(window_length_seconds * long_recording.sampling_frequency)
            overlap_samples = int(window_overlap_seconds * long_recording.sampling_frequency)
            num_windows = int((N - window_size_samples) / (window_size_samples - overlap_samples)) + 1
            return num_windows

        def _window_recording(recording, window_length_sec, window_overlap):
            num_windows = _calc_num_windows(recording, window_length_sec, window_overlap)
            recording_windows = []
            for i in range(num_windows):
                start = i * (window_length_sec - window_overlap)
                end = start + window_length_sec
                recording_window = recording.crop(start, end)
                recording_windows.append(Recording(recording_window.values, recording.sampling_frequency,
                                                   recording.name + '_window_' + str(i)))
            return recording_windows

        samples = []
        num_windows = _calc_num_windows(long_recording, window_length_seconds, window_overlap_seconds)
        if labels is not None:
            if len(labels) != num_windows:
                raise Warning("Number of labels does not match number of windows")
                labels = None
        for i, w in enumerate(_window_recording(long_recording, window_length_seconds, window_overlap_seconds)):
            if limit_num_windows and i > limit_num_windows:
                print(f"reached limit of {limit_num_windows} windows")
                break
            print(f"creating sample of window {i} out of {num_windows}")
            sample = Sample(w, w.name, labels[i] if labels is not None else None)
            samples.append(sample)
        self.samples_list.extend(samples)

    def add_samples(self, samples):
        if isinstance(samples, list):
            self.samples_list.extend(samples)
        else:
            self.samples_list.append(samples)

    def extract_features_all_samples(self):
        for i, sample in enumerate(self.samples_list):
            if i < len(self.stream_features_list):
                continue
            self.samples_features_list.append(sample.extract_features())

    def predict_stream(self): # placeholder
        return None

    def remove_recordings(self):
        self.recording = None
        for sample in self.samples_list:
            sample.remove_recording()

    def remove_samples(self, sample_num):
        for i in sample_num:
            self.samples_list.pop(i)

    def add_previous_samples_features(self, num_lookback_samples):
        for i, sample in enumerate(self.samples_list):
            feats = {}
            if i > num_lookback_samples:
                for j in range(1, num_lookback_samples+1):
                    feats[f'pitch_frequency_lookback_{j}'] = self.samples_list[i-j].features["pitch_frequency"]
                    feats[f'hps_sparsity_lookback_{j}'] = self.samples_list[i-j].features["hps_sparsity"]
                    feats[f'pitch_strength_lookback_{j}'] = self.samples_list[i-j].features["pitch_strength"]
                sample.add_features(feats)
        self.samples_list = self.samples_list[num_lookback_samples+1:]


