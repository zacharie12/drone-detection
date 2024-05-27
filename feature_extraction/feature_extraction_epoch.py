import numpy as np
from numpy.linalg import norm

def sparsity(x):
    """ returns relative sparsity of a spectrum"""
    n = len(x)
    norm_l1 = norm(x, 1)
    norm_l2 = norm(x)
    sparsity_score = (np.sqrt(n) - norm_l1 / norm_l2) / (np.sqrt(n) - 1)
    return sparsity_score


def get_hps_features(epoch):
    hps_freqs = epoch.hps.x
    hps_values = epoch.hps.y
    hps_max_freq_pitch_search = 180
    freq = hps_freqs[np.argmax(hps_values[:hps_max_freq_pitch_search])]
    sparsity_score = sparsity(hps_values)
    strength = np.max(hps_values)
    return freq, sparsity_score, strength


class FeatureExtractionEpoch(object):
    def __init__(self, epoch, previous_epochs=None):
        self.epoch = epoch
        self.previous_epochs = previous_epochs
        self.features = {}

    def __str__(self):
        return "Epoch: " + str(self.epoch) + " Features: " + str(self.features)

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



    def extract_features(self):
        self.features['full_bandwidth_rms'] = self.epoch.recording.audio_data.rms()
        self.features['bandpassed_rms'] = self.epoch.bandpassed_recording.audio_data.rms()
        self.features["1500_to_max_rms"] = self.epoch.recording.psd.frequency_band_rms(1500)
        freq, spars, strgth = self.get_hps_features(self.epoch)
        self.features["pitch_frequency"] = freq
        self.features["hps_sparsity"] = spars
        self.features["pitch_strength"] = strgth
        if self.previous_epochs is not None:
            self.extract_previous_epoch_features()

    def extract_previous_epoch_features(self):
        for i, epoch in enumerate(self.previous_epochs):
            freq, spars, strgth = get_hps_features(epoch)
            self.features[f"previous_epoch_pitch_frequency_{i}"] = freq
            self.features[f"previous_epoch_hps_sparsity_{i}"] = spars
            self.features[f"previous_epoch_pitch_strength_{i}"] = strgth