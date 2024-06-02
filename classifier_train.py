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
from models import *

if __name__ == '__main__':
    with open('/Users/zachariecohen/drone-detection/dataset/labeled_datasets/final_dataset_oc_2024-06-02 13:31:20.603969.pkl', 'rb') as file:
        dataset = pickle.load(file)
    dataset.dataframe = dataset.dataframe.dropna().reset_index(drop=True)
    # Separate the features and the labels
    X = dataset.dataframe.drop('label', axis=1)
    y = dataset.dataframe['label']

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Create an SVM model with a polynomial kernel and specific hyperparameters
    svm = CustomSVM(kernel='poly', degree=3, gamma='scale', coef0=1, C=1.0).get_model()

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

    # Find the index of the best number of features based on maximum accuracy
    best_index = np.argmax(performance_dict['accuracy'])
    best_num_features = performance_dict['num_features'][best_index]
    best_accuracy = performance_dict['accuracy'][best_index]
    best_recall = performance_dict['recall'][best_index]
    best_precision = performance_dict['precision'][best_index]
    best_features = sorted_indices[:best_num_features]

    # Print the results
    print(f"Best number of features: {best_num_features}")
    print(f"Indices of best features: {best_features}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Best recall: {best_recall:.4f}")
    print(f"Best precision: {best_precision:.4f}")

    # Plot t-SNE graph with the best number of features
    best_num_features = np.argmax(performance_dict['accuracy'])
    best_features = sorted_indices[:best_num_features]

    # t-SNE transformation
    tsne = TSNE(n_components=2, random_state=42)
    X_train_best_tsne = tsne.fit_transform(X_train_np[:, best_features])

    # Plot the t-SNE results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_best_tsne[:, 0], X_train_best_tsne[:, 1], c=y_train_np, cmap='viridis', marker='o')
    plt.colorbar()
    plt.title('t-SNE plot of the best features')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.show()