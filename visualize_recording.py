from recording_class.recording import Recording
from moviepy.editor import VideoFileClip
import sounddevice as sd
import librosa as lr
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
    path = "/Users/ocarmi/Documents/private/Aurras/drone-detection/OSINT/OSINT/Red/Ababil T/7may eretzjihad.mp4"
    recording = from_mp4(path)
    recording.crop(0.28, 4)
    recording.visualize()
    recording.play_audio()
    a=5

