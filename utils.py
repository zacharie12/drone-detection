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
        video.audio.write_audiofile(file_path[:-1] + "3") # save to mp3
        audio, sampling_frequency = lr.load(file_path[:-1] + "3") # read mp3 with librosa
    name = file_path.split('/')[-1].split('.')[0]  # Extract filename without extension as name
    return Recording(audio, sampling_frequency, name)