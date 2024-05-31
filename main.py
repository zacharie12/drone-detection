from utils import from_mp4
from dataset.dataset import Dataset
import os
from stream.stream import Stream
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score
from skrebate import ReliefF
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

sample_window_duration_sec = 0.5
window_ovelap_sec = 0.25
limit_num_windows = False


if __name__ == '__main__':
    streams = []
    directory = "/Users/ocarmi/Documents/private/Aurras/drone_detection/OSINT/OSINT/Red/Ababil T" #red
    # directory = "/Users/ocarmi/Documents/private/Aurras/drone_detection/OSINT/OSINT/Blue" # blue

    # directory = "/Users/zachariecohen/Desktop/drone-detection/Red/Ababil T" #red
    # directory = "/Users/zachariecohen/Desktop/drone-detection/OSINT/OSINT/Blue" # blue
    labeled_files = []
    for file in os.listdir(directory):
        if file.endswith(".mp4"):
            path = os.path.join(directory, file)
            recording = from_mp4(path)
            if '20nov' in file:
                continue
            print(f"finished loading {file}")
            stream = Stream(recording=recording,
                            sample_window_duration_sec=sample_window_duration_sec,
                            window_ovelap_sec=window_ovelap_sec,
                            limit_num_windows=limit_num_windows,
                            name=path.split('/')[-1].split('.')[0])
            stream.manually_label_samples()
            labeled_files.append(file)
            streams.append(stream)
            with open(f'{file}_stream_class_with_labels.pkl', 'wb') as f:
                pickle.dump(stream, f)
            print(f"finished creating stream for {file}")
    dataset = Dataset(streams)
    dataset.remove_recordings()
    with open('dataset_oc1.pkl', 'wb') as f:
        pickle.dump(dataset, f)

dataset.dataframe['label'] = np.random.choice([0, 1], size=len(dataset.dataframe))
dataset.dataframe = dataset.dataframe.dropna().reset_index(drop=True)
# Separate the features and the labels
X = dataset.dataframe.drop('label', axis=1)
y = dataset.dataframe['label']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize SVM classifier
svm = SVC(kernel='linear', probability=True)

# Initialize ReliefF
relief = ReliefF()

# Convert to NumPy arrays
X_train_np = np.array(X_train)
y_train_np = np.array(y_train)

# Apply ReliefF feature selection using NumPy arrays
relief.fit(X_train_np, y_train_np)

# Get feature scores and sort them
feature_scores = relief.feature_importances_
sorted_indices = np.argsort(feature_scores)[::-1]

# Dictionary to hold accuracy, recall, and precision for each feature set size
performance_dict = {'num_features': [], 'accuracy': [], 'recall': [], 'precision': []}

# Evaluate SVM with different number of top features selected by ReliefF
for num_features in range(1, len(X.columns) + 1):
    top_features = sorted_indices[:num_features]

    # Train the SVM classifier with the selected features
    svm.fit(X_train[:, top_features], y_train)

    # Predict on the test set with the selected features
    y_pred = svm.predict(X_test[:, top_features])

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    # Save the performance metrics
    performance_dict['num_features'].append(num_features)
    performance_dict['accuracy'].append(accuracy)
    performance_dict['recall'].append(recall)
    performance_dict['precision'].append(precision)

# Plot the performance metrics
plt.figure(figsize=(10, 6))
plt.plot(performance_dict['num_features'], performance_dict['accuracy'], label='Accuracy')
plt.plot(performance_dict['num_features'], performance_dict['recall'], label='Recall')
plt.plot(performance_dict['num_features'], performance_dict['precision'], label='Precision')
plt.xlabel('Number of Features')
plt.ylabel('Performance')
plt.legend()
plt.title('SVM Performance with Varying Number of Features')
plt.show()

# Plot t-SNE graph with the best number of features
best_num_features = np.argmax(performance_dict['accuracy'])
best_features = sorted_indices[:best_num_features]

# t-SNE transformation
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne