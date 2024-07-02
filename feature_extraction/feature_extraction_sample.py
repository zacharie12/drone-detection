import numpy as np
from numpy.linalg import norm


def sparsity(x):
    n = len(x)
    norm_l1 = norm(x, 1)
    norm_l2 = norm(x)
    sparsity_score = (np.sqrt(n) - norm_l1 / norm_l2) / (np.sqrt(n) - 1)
    return sparsity_score





class FeatureExtractionSample(object):
    def __init__(self, epoch):
        self.epoch = epoch
        self.features = {}

    def __str__(self):
        return "Sample: " + str(self.epoch) + " Features: " + str(self.features)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.epoch == other.epoch and self.features == other.features

    def get_epoch(self):
        return self.epoch

    def get_features(self):
        return self.features

    def get_feature(self, feature_name):
        return self.features[feature_name]

    def get_feature_names(self):
        return self.features.keys()

    def calc_hps_features(self):
        hps_freqs = self.epoch.hps.x
        hps_values = self.epoch.hps.y
        hps_max_freq_pitch_search = 180
        hps_max_freq_pitch_index = int(hps_max_freq_pitch_search / hps_freqs[1])
        freq = hps_freqs[np.argmax(hps_values[:hps_max_freq_pitch_index])]
        sparsity_score = sparsity(hps_values)
        strength = np.max(hps_values[:hps_max_freq_pitch_index]) / np.sum(hps_values[:hps_max_freq_pitch_index])
        return freq, sparsity_score, strength

    def extract_features(self):
        self.features['full_bandwidth_rms'] = self.epoch.recording.audio_data.rms()
        self.features['bandpassed_rms'] = self.epoch.bandpassed_recording.audio_data.rms()
        self.features['bandpassed_crest_factor'] = self.epoch.bandpassed_recording.audio_data.crest_factor()
        self.features["1500_to_max_rms"] = self.epoch.recording.psd.frequency_band_rms(1500)
        freq, spars, strgth = self.calc_hps_features()
        self.features["pitch_frequency"] = freq
        self.features["hps_sparsity"] = spars
        self.features["pitch_strength"] = strgth
        for i in range(100, 1000, 100):
            self.features[f"{i}hz_{i + 100}hz_rms"] = self.epoch.recording.psd.frequency_band_rms(i, i + 100)
        return self.features



