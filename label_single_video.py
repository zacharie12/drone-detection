from utils import from_mp4, label_single_video
import pickle
sample_window_duration_sec = 0.5
window_ovelap_sec = 0.25
limit_num_windows = False
import datetime

if __name__ == '__main__':
    path = input("Enter the full path of the video file: ")
    most_recent_dataset = input("Enter the full path of the most recent dataset: ")
    initials = input("Enter your initials for the output file name:")
    stream = label_single_video(path, initials, sample_window_duration_sec, window_ovelap_sec, limit_num_windows)
    dataset = pickle.load(open(most_recent_dataset, 'rb'))
    dataset.add_stream(stream)
    dataset.make_dataframe_of_feats_and_labels()
    with open(f'dataset/labeled_datasets/final_dataset_{initials}_{datetime.datetime.now()}.pkl', 'wb') as f:
        pickle.dump(dataset, f)