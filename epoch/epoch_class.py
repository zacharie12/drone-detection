from drone_detection.preprocessing.preprocessing import bpf, HPS
from drone_detection.recording_class.recording import Recording, Vector

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



