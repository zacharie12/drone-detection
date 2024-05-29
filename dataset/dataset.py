import numpy as np

class Dataset():

    def __init__(self, streams):
        # stream class has the following attributes:
        # - epoch_features_list: list of features for each epoch
        # - labels: list of labels for each epoch
        # - epochs_list: list of epochs (raw audio data)
        self.streams = streams # list of streams (each stream is linked epochs)
        self.all_features_concatenated = self.concatenate_features() # list of all features concatenated from all streams
        self.all_labels_concatenated = self.concatenate_labels() # list of all labels concatenated from all streams

    def __len__(self):
        return len(self.stream)

    def add_stream(self, stream):
        self.streams.append(stream)

    def concatenate_labels(self):
        return np.concatenate(self.streams.labels)

    def concatenate_features(self):
        return [item for stream in self.streams for item in stream.epoch_features_list]


