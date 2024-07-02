from utils import from_mp4, label_single_video
import pickle
sample_window_duration_sec = 0.5
window_ovelap_sec = 0.25
limit_num_windows = False
import datetime
import os

if __name__ == '__main__':
    path = input("Enter the full path of the video file: ")
    most_recent_dataset_path = input("Enter the full path of the most recent dataset: ")
    initials = input("Enter your initials for the output file name:")
    dataset = pickle.load(open(most_recent_dataset_path, 'rb'))
    labeled_videos = [stream.name for stream in dataset.streams]
    if path.split('/')[-1].split('.')[0] in labeled_videos:
        print("This video has already been labeled.")
    else:
        stream = label_single_video(path, initials, sample_window_duration_sec, window_ovelap_sec, limit_num_windows)
        dataset.add_stream(stream)
        dataset.make_dataframe_of_feats_and_labels()
        file_name = f'final_dataset_{initials}_{datetime.datetime.now()}.pkl'.replace('-', '_').replace(':', '_').replace(' ', '_').replace('.', '_')
        path_to_save = os.path.join('dataset', 'labeled_datasets', file_name)
        with open(path_to_save, 'wb') as f:
            pickle.dump(dataset, f)