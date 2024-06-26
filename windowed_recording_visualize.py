from recording_class.recording import Recording
from sample.sample import Sample
from moviepy.editor import VideoFileClip
import sounddevice as sd
import librosa as lr
import os
import pandas as pd
import pickle


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
    epochs = []
    directory = "/Users/ocarmi/Documents/private/Aurras/drone_detection/OSINT/OSINT/Red/Ababil T" #red
    video_cropping_times = pd.read_csv(
        "/Users/ocarmi/Documents/private/Aurras/drone_detection/OSINT/OSINT/Red/Ababil T/video_cropping_times.csv")  # red
    # directory = "/Users/ocarmi/Documents/private/Aurras/drone_detection/OSINT/OSINT/Blue" # blue
    # video_cropping_times = pd.read_csv("/Users/ocarmi/Documents/private/Aurras/drone_detection/OSINT/OSINT/Blue/video_cropping_times.csv") # blue
    for file in os.listdir(directory):
        if file.endswith(".mp4") and video_cropping_times['video_name'].str.contains(file[:-4]).any() and 'tal' not in file and '20nov' not in file:
            path = os.path.join(directory, file)
            recording = from_mp4(path)
            row = video_cropping_times.loc[video_cropping_times['video_name']==file[:-4]]
            recording.crop(row["start_sec"].values[0], row["end_sec"].values[0])
            epoch = Sample(recording, path.split('/')[-1].split('.')[0])
            epoch.preprocess()
            epoch.crop(row["start_sec"].values[0], row["end_sec"].values[0])
            epoch.window_recording(use_cropped=False)
            epoch.windows_hps()
            epoch.visualize_windows(max_freq=2000)
            a = 5
    a = 5
    # with open('/drone_detection/OSINT/OSINT/Red/Ababil T/recordings.pickle', 'wb') as f:
    #     pickle.dump(recordings, f)

