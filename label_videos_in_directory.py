from utils import label_single_video
import pickle
import os
sample_window_duration_sec = 0.5
window_ovelap_sec = 0.25
limit_num_windows = False
import datetime

if __name__ == '__main__':
    streams = []
    dir = input("Enter the directory full path: ")
    most_recent_dataset_path = input("Enter the full path of the final dataset: ")
    dataset = pickle.load(open(most_recent_dataset_path, 'rb'))
    labeled_videos = [stream.name for stream in dataset.streams]
    initials = input("Enter your initials for the output file name:")
    for file in os.listdir(dir):
        if file.endswith('.mp4'):
            path = os.path.join(dir, file)
            if path.split('/')[-1].split('.')[0] in labeled_videos:
                print("This video has already been labeled.")
            else:
                stream = label_single_video(path, initials, sample_window_duration_sec, window_ovelap_sec, limit_num_windows)
                streams.append(stream)
    for stream in streams:
        dataset.add_stream(stream)
    dataset.make_dataframe_of_feats_and_labels()
    with open(f'dataset/labeled_datasets/final_dataset_{initials}_{datetime.datetime.now()}.pkl', 'wb') as f:
        pickle.dump(dataset, f)