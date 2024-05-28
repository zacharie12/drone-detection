from drone_detection.epoch.epoch_class import Epoch
from drone_detection.feature_extraction.feature_extraction_epoch import FeatureExtractionEpoch


class Stream():
    def __init__(self):
        self.epochs_list = []
        self.epoch_features_list = []

    def add_epochs(self, epochs):
        self.epochs_list.extend(epochs)

    def extract_features(self):
        for i, epoch in enumerate(self.epochs_list):
            if i > 1:
                previous_epochs = self.epochs_list[i-2:i]
            else:
                previous_epochs = None
            epoch_features_class = FeatureExtractionEpoch(epoch, previous_epochs)
            epoch_features_class.extract_features()
            self.epoch_features_list.append(epoch_features_class.get_features())