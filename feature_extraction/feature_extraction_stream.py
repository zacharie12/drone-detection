

class FeatureExtractionStream():

    def __init__(self, stream):
        self.stream = stream
        self.features = self.extract_features()