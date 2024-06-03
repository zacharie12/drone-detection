from recording_class.recording import Recording
from sample.sample import Sample
from moviepy.editor import VideoFileClip
import librosa as lr
import plotly.graph_objects as go
import os
import pandas as pd
import pickle
from stream.stream import Stream
from sklearn.manifold import TSNE
import numpy as np


def from_mp4(file_path):
    video = VideoFileClip(file_path)
    try:
        audio, sampling_frequency = lr.load(file_path[:-1] + "3")
    except:
        video.audio.write_audiofile(file_path[:-1] + "3")  # save to mp3
        audio, sampling_frequency = lr.load(file_path[:-1] + "3")  # read mp3 with librosa
    name = file_path.split('/')[-1].split('.')[0]  # Extract filename without extension as name
    return Recording(audio, sampling_frequency, name)

def make_stream_with_labels(path, initials,
                            sample_window_duration_sec=0.5,
                            window_ovelap_sec=0.25,
                            limit_num_windows=False,
                            labels=None):
    file = os.path.basename(path)
    recording = from_mp4(path)
    print(f"finished loading {file}")
    stream = Stream(recording=recording,
                    sample_window_duration_sec=sample_window_duration_sec,
                    window_ovelap_sec=window_ovelap_sec,
                    limit_num_windows=limit_num_windows,
                    name=file,
                    labels=labels)
    stream.remove_recordings()
    with open(f'{file}_stream_class_with_labels_{initials}', 'wb') as f:
        pickle.dump(stream, f)
    print(f"finished creating & saving stream for {file}")
    return stream

def label_single_video(path, initials,
                       sample_window_duration_sec=0.5,
                       window_ovelap_sec=0.25,
                       limit_num_windows=False):
    file = os.path.basename(path)
    recording = from_mp4(path)
    print(f"finished loading {file}")
    stream = Stream(recording=recording,
                    sample_window_duration_sec=sample_window_duration_sec,
                    window_ovelap_sec=window_ovelap_sec,
                    limit_num_windows=limit_num_windows,
                    name=file)
    stream.manually_label_samples()
    # for sample in stream.samples_list:
    #     sample.label = 0
    stream.remove_recordings()
    with open(f'{file}_stream_class_with_labels_{initials}', 'wb') as f:
        pickle.dump(stream, f)
    print(f"finished creating & saving stream for {file}")
    return stream

