from sklearn.svm import SVC


class CustomSVM:
    def __init__(self, kernel, **kwargs):
        """
        Custom SVM class constructor.

        Parameters:
        - kernel: str, the kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        - kwargs: additional keyword arguments for hyperparameters

        Example usage:
        svm_model = CustomSVM(kernel='poly', degree=3, gamma='scale', coef0=1, C=1.0)
        svm_model = CustomSVM(kernel='rbf', gamma=0.1, C=1.0)
        """
        self.kernel = kernel
        self.kwargs = kwargs
        self.model = None

    def get_model(self):
        """
        Method to get the configured SVM model.

        Returns:
        - an instance of SVC with the specified kernel and hyperparameters
        """
        # Create the SVM model with the specified kernel and hyperparameters
        self.model = SVC(kernel=self.kernel, probability=True, **self.kwargs)
        return self.model