import os
from moviepy.editor import AudioFileClip
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np


def extract_audio_from_mp4(dir_path):
    audio_files_dict = {}
    # Iterate over all files in directory and subdirectories
    for subdir, dirs, files in os.walk(dir_path):
        # Get the name of the current folder
        folder_name = os.path.basename(subdir)
        # Initialize the list for this folder in the dictionary if not already present
        if folder_name not in audio_files_dict:
            audio_files_dict[folder_name] = []
        for file in files:
            # Check if the file is an mp4
            if file.endswith('.mp4'):
                filepath = os.path.join(subdir, file)
                # Initialize AudioFileClip directly with the video file
                audio_clip = AudioFileClip(filepath)
                # Define the output audio file path
                audio_file_path = os.path.splitext(filepath)[0] + '.wav'
                # Save the audio file to the same directory with the same name, but with .wav extension
                audio_clip.write_audiofile(audio_file_path)
                # Append the audio file path to the corresponding list in the dictionary
                audio_files_dict[folder_name].append(audio_file_path)

                # Return the dictionary containing the lists of audio files
    return audio_files_dict


dir_path = 'OSINT'  # replace with your directory path
audio_files_dict = extract_audio_from_mp4(dir_path)

def plot_frequency_domain(audio_files_dict):
    for folder, audio_files in audio_files_dict.items():
        print(f"Processing folder: {folder}")
        for audio_file in audio_files:
            # Read the audio file
            sample_rate, data = wavfile.read(audio_file)
            # Check if the audio file is mono or stereo and take the first channel if it's stereo
            if len(data.shape) > 1:
                data = data[:, 0]
                # Calculate the duration of the audio file
            duration = len(data) / sample_rate
            # Perform the Fourier transform
            fft_output = np.fft.fft(data)
            # Get the power spectrum
            power_spectrum = np.abs(fft_output) ** 2
            # Create the frequency axis
            freq_axis = np.fft.fftfreq(len(fft_output), 1 / sample_rate)

            # Plot the frequency domain representation
            plt.figure(figsize=(10, 6))
            plt.plot(freq_axis[:len(fft_output) // 2], power_spectrum[:len(fft_output) // 2])
            plt.title(f"Frequency Domain Plot for {os.path.basename(audio_file)}")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power")
            plt.xlim(0, sample_rate // 2)  # Limit x-axis to half the sample rate (Nyquist frequency)
            plt.show()

        # Assuming `audio_files_dict` is the dictionary with your audio file paths


plot_frequency_domain(audio_files_dict)
