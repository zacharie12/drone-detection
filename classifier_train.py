import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from skrebate import ReliefF
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from models import *
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Define hyperparameters for SVM model at the beginning of the script
SVM_PARAMS = {
    'C': 1.0,  # Regularization parameter
    'kernel': 'rbf',  # Kernel type: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable
    'degree': 3,  # Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.
    'gamma': 'scale',  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
    'coef0': 0.0,  # Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
    'random_state': 42
}

# Define hyperparameters for each model at the beginning of the script
MLP_PARAMS = {
    'input_size': None,  # to be set after data is loaded
    'hidden_layers': [64, 32],
    'output_size': 1
}
LSTM_PARAMS = {
    'input_size': None,  # to be set after data is loaded
    'hidden_size': 128,
    'num_layers': 2,
    'output_size': 1
}
GRU_PARAMS = {
    'input_size': None,  # to be set after data is loaded
    'hidden_size': 128,
    'num_layers': 2,
    'output_size': 1
}

RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'random_state': 42
}
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.1,
    'random_state': 42
}

# Define training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001


# Helper function to calculate metrics for PyTorch models
def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    accuracy = accuracy_score(y_true, y_pred > 0.5)
    precision = precision_score(y_true, y_pred > 0.5, zero_division=0)
    recall = recall_score(y_true, y_pred > 0.5, zero_division=0)
    return accuracy, precision, recall


# Function to convert data to PyTorch DataLoader
def get_data_loader(X, y, batch_size):
    tensor_x = torch.Tensor(X)  # transform to torch tensor
    tensor_y = torch.Tensor(y.values)  # convert the pandas Series to numpy array before creating the tensor
    dataset = TensorDataset(tensor_x, tensor_y)  # create your dataset
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Function to train PyTorch models
def train_pytorch_model(model, train_loader, val_loader, num_epochs, learning_rate):
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            # Validation phase
        model.eval()
        with torch.no_grad():
            val_labels = []
            val_outputs = []
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_labels.append(labels)
                val_outputs.append(outputs)
            val_labels = torch.cat(val_labels)
            val_outputs = torch.cat(val_outputs)
            val_accuracy, val_precision, val_recall = calculate_metrics(val_labels, val_outputs)
            print(
                f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')


# Function to evaluate PyTorch models on a test set
def test_pytorch_model(model, test_loader):
    criterion = torch.nn.BCEWithLogitsLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Test phase
    model.eval()
    test_labels = []
    test_outputs = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_labels.append(labels)
            test_outputs.append(outputs)

    test_labels = torch.cat(test_labels)
    test_outputs = torch.cat(test_outputs)
    test_loss = criterion(test_outputs.squeeze(), test_labels.float())

    test_accuracy, test_precision, test_recall = calculate_metrics(test_labels, test_outputs)
    print(
        f'Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}')

    return test_loss.item(), test_accuracy, test_precision, test_recall


def train_evaluate_sklearn_model_with_relief(model, X_train, y_train, X_test, y_test):
    performance_dict = {'num_features': [], 'accuracy': [], 'recall': [], 'precision': []}

    # Apply ReliefF feature selection
    relief = ReliefF()
    relief.fit(X_train, y_train.to_numpy())  # Convert y_train to numpy array here

    # Get feature scores and sort them
    feature_scores = relief.feature_importances_
    sorted_indices = np.argsort(feature_scores)[::-1]

    # Evaluate model with different number of top features selected by ReliefF
    for num_features in range(1, len(X_train[0]) + 1):
        top_features = sorted_indices[:num_features]
        X_train_relief = X_train[:, top_features]
        X_test_relief = X_test[:, top_features]

        # Train the model with the selected features
        model.fit(X_train_relief, y_train)

        # Predict on the test set with the selected features
        y_pred = model.predict(X_test_relief)

        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)

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
    plt.title('Model Performance with Varying Number of Features')
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

    # Plot t-SNE graph with the best number of features
    tsne = TSNE(n_components=2, random_state=42)
    X_train_best_tsne = tsne.fit_transform(X_train[:, best_features])

    # Plot the t-SNE results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_best_tsne[:, 0], X_train_best_tsne[:, 1], c=y_train, cmap='viridis', marker='o')
    plt.colorbar()
    plt.title('t-SNE plot of the best features')
    plt.xlabel('t-SNE feature')
    return best_accuracy, best_recall, best_precision


if __name__ == '__main__':

    model_name = 'mlp'  # Specify the model name here ('svm', 'randomforest', 'xgboost', 'mlp', 'lstm', 'gru')
    with open('/Users/zachariecohen/drone-detection/dataset/labeled_datasets/final_dataset_oc_2024-06-02 13:31:20.603969.pkl', 'rb') as file:
        dataset = pickle.load(file)
    dataset.dataframe = dataset.dataframe.dropna().reset_index(drop=True)
    # Separate the features and the labels
    X = dataset.dataframe.drop('label', axis=1)
    y = dataset.dataframe['label'].reset_index(drop=True)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Check if the model requires a validation set
    models_requiring_val_set = ['mlp', 'lstm', 'gru']
    if model_name.lower() in models_requiring_val_set:
        # Split the dataset into training, validation, and test sets
        X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125,
                                                          random_state=42)  # 0.125 x 0.8 = 0.1
    else:
        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        X_val, y_val = None, None

    if model_name.lower() in ['mlp', 'lstm', 'gru']:
        # For PyTorch models, create DataLoaders for train, val, and test sets
        train_loader = get_data_loader(X_train, y_train, BATCH_SIZE)
        val_loader = get_data_loader(X_val, y_val, BATCH_SIZE)
        test_loader = get_data_loader(X_test, y_test, BATCH_SIZE)
        # Create a PyTorch model based on the model_name
        if model_name.lower() == 'mlp':
            MLP_PARAMS['input_size'] = X_train.shape[1]
            model = CustomMLP(**MLP_PARAMS)
        elif model_name.lower() == 'lstm':
            LSTM_PARAMS['input_size'] = X_train.shape[1]
            model = CustomLSTM(**LSTM_PARAMS)
        elif model_name.lower() == 'gru':
            GRU_PARAMS['input_size'] = X_train.shape[1]
            model = CustomGRU(**GRU_PARAMS)

            # Train the PyTorch model
        train_pytorch_model(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE)

        # Test the PyTorch model
        test_pytorch_model(model, test_loader)

    # Models that are not deep
    elif model_name.lower() == 'randomforest':
        model = RandomForestClassifier(**RF_PARAMS)
        accuracy, recall, precision = train_evaluate_sklearn_model_with_relief(model, X_train, y_train, X_test, y_test)
        print(f"Random Forest: Accuracy = {accuracy:.4f}, Recall = {recall:.4f}, Precision = {precision:.4f}")
    elif model_name.lower() == 'xgboost':
        model = XGBClassifier(**XGB_PARAMS)
        accuracy, recall, precision = train_evaluate_sklearn_model_with_relief(model, X_train, y_train, X_test, y_test)
        print(f"XGBoost: Accuracy = {accuracy:.4f}, Recall = {recall:.4f}, Precision = {precision:.4f}")
    elif model_name.lower() == 'svm':
        model = SVC(**SVM_PARAMS)
        accuracy, recall, precision = train_evaluate_sklearn_model_with_relief(model, X_train, y_train, X_test, y_test)
        print(f"SVM: Accuracy = {accuracy:.4f}, Recall = {recall:.4f}, Precision = {precision:.4f}")




