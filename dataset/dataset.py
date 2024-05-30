import numpy as np
import pandas as pd


class Dataset():

    def __init__(self, streams):
        # stream class has the following attributes:
        # - stream_features_list: list of features for each sample
        # - labels: list of labels for each sample
        # - samples_list: list of epochs (raw audio data)
        self.streams = streams # list of streams (each stream is linked epochs)
        self.dataframe = self.make_dataframe_of_feats_and_labels()

    def __len__(self):
        return len(self.streams)

    def add_stream(self, stream):
        self.streams.append(stream)

    def make_dataframe_of_feats_and_labels(self):
        data = []
        for stream in self.streams:
            for i, samples in enumerate(stream.samples_list):
                data.append(list(samples.features.values()) + [samples.label])
        columns = list(stream.samples_list[0].features.keys()) + ['label']
        return pd.DataFrame(data, columns=columns)

    def remove_recordings(self):
        for stream in self.streams:
            stream.remove_recordings()



