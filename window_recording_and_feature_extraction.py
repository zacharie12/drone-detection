from recording_class.recording import Recording
from epoch.epoch_class import Epoch
from moviepy.editor import VideoFileClip
import librosa as lr
import plotly.graph_objects as go
import os
import pandas as pd
import pickle
from stream.stream import Stream
from sklearn.manifold import TSNE
import numpy as np


def visualize_tsne(features_list):
    # Convert features to a 2D array
    features_array = np.array([list(epoch.values()) for epoch in features_list])

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=features_array.shape[0] / 3)
    embedded_features = tsne.fit_transform(features_array)

    # Plot the embedded features
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=embedded_features[:, 0], y=embedded_features[:, 1], mode='markers'))
    fig.update_layout(title_text='t-SNE Visualization of Epoch Features')
    fig.show(renderer='browser')


def window_recording(recording, window_length_sec=0.5):
    recording_windows = []
    num_windows = int(recording.audio_data.length / (window_length_sec * recording.sampling_frequency))
    for i in range(num_windows):
        start = i * window_length_sec
        end = (i + 1) * window_length_sec
        recording_window = recording.crop(start, end)
        recording_windows.append(Recording(recording_window.values, recording.sampling_frequency, recording.name + '_window_' + str(i)))
    return recording_windows


def from_mp4(file_path):
    video = VideoFileClip(file_path)
    try:
        audio, sampling_frequency = lr.load(file_path[:-1] + "3")
    except:
        video.audio.write_audiofile(file_path[:-1] + "3") # save to mp3
        audio, sampling_frequency = lr.load(file_path[:-1] + "3") # read mp3 with librosa
    name = file_path.split('/')[-1].split('.')[0]  # Extract filename without extension as name
    return Recording(audio, sampling_frequency, name)


if __name__ == '__main__':
    streams = []
    directory = "/Users/ocarmi/Documents/private/Aurras/drone_detection/OSINT/OSINT/Red/Ababil T" #red
    # directory = "/Users/ocarmi/Documents/private/Aurras/drone_detection/OSINT/OSINT/Blue" # blue
    for file in os.listdir(directory):
        if file.endswith(".mp4"):
            path = os.path.join(directory, file)
            recording = from_mp4(path)
            print(f"finished loading {file}")
            stream = Stream()
            epochs = []
            window_length_seconds = 0.5
            num_windows = int(recording.audio_data.length / (window_length_seconds * recording.sampling_frequency))
            for i, w in enumerate(window_recording(recording, window_length_seconds)):
                print(f"creating epoch of window {i} out of {num_windows}")
                epoch = Epoch(w, path.split('/')[-1].split('.')[0])
                epoch.preprocess()
                epoch.calc_hps()
                epochs.append(epoch)
            stream.add_epochs(epochs)
            print(f"finished creating epochs for {file}, starting feature extraction")
            stream.extract_features_all_epochs()
            # visualize_tsne(stream.epoch_features_list[2:])
            print(f"finished feature extraction for {file}")
            streams.append(stream)
            a = 5
    a = 5
    # with open('/drone_detection/OSINT/OSINT/Red/Ababil T/recordings.pickle', 'wb') as f:
    #     pickle.dump(recordings, f)

