from drone_detection.utils import from_mp4
from dataset.dataset import Dataset
import os
from stream.stream import Stream
sample_window_duration_sec = 0.5
window_ovelap_sec = 0.25
limit_num_windows = 50

if __name__ == '__main__':
    streams = []
    directory = "/Users/ocarmi/Documents/private/Aurras/drone_detection/OSINT/OSINT/Red/Ababil T" #red
    # directory = "/Users/ocarmi/Documents/private/Aurras/drone_detection/OSINT/OSINT/Blue" # blue
    for file in os.listdir(directory):
        if file.endswith(".mp4"):
            path = os.path.join(directory, file)
            recording = from_mp4(path)
            print(f"finished loading {file}")
            stream = Stream(recording=recording,
                            sample_window_duration_sec=sample_window_duration_sec,
                            window_ovelap_sec=window_ovelap_sec,
                            limit_num_windows=limit_num_windows,
                            name=path.split('/')[-1].split('.')[0])
            stream.manually_label_samples()
            streams.append(stream)
            print(f"finished creating stream for {file}")
            a = 5
    dataset = Dataset(streams)
    a = 5
