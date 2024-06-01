from utils import from_mp4
from dataset.dataset import Dataset
import os
from stream.stream import Stream
import pickle
sample_window_duration_sec = 0.5
window_ovelap_sec = 0.25
limit_num_windows = False
import datetime

if __name__ == '__main__':
    streams = []
    path = input("Enter the full path of the file: ")
    file = os.path.basename(path)
    recording = from_mp4(path)
    print(f"finished loading {file}")
    stream = Stream(recording=recording,
                    sample_window_duration_sec=sample_window_duration_sec,
                    window_ovelap_sec=window_ovelap_sec,
                    limit_num_windows=limit_num_windows,
                    name=file)
    stream.manually_label_samples()
    stream.remove_recordings()
    streams.append(stream)
    initials = input("Enter your initials for the output file name: ")
    with open(f'{file}_stream_class_with_labels_{initials}', 'wb') as f:
        pickle.dump(stream, f)
    print(f"finished creating stream for {file}")
    dataset = pickle.load(open('dataset/labeled_datasets/final_dataset.pkl', 'rb'))
    dataset.add_stream(stream)
    dataset.make_dataframe_of_feats_and_labels()
    with open(f'dataset/labeled_datasets/final_dataset_{initials}_{datetime.datetime.now()}.pkl', 'wb') as f:
        pickle.dump(dataset, f)