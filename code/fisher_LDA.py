import numpy as np
from numpy import linalg as LA

class Fisher_LDA:

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.U = None

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        X_transformed = self.transform(X, Y)
        return X_transformed

    def fit(self, X, Y):
        # X: rows are features and columns are samples
        # Y: rows are features and columns are samples
        mean_total = X.mean(axis=1)
        n_dimensions = X.shape[0]
        X_separated_classes = Fisher_LDA.separate_samples_of_classes(X=X.T, y=Y.T)
        n_classes = len(X_separated_classes)
        S_b = np.zeros((n_dimensions, n_dimensions))
        S_w = np.zeros((n_dimensions, n_dimensions))
        for class_index in range(n_classes):
            class_samples = X_separated_classes[class_index].T
            mean_class = class_samples.mean(axis=1)
            n_class_samples = class_samples.shape[1]
            a = (mean_class - mean_total).reshape((-1,1))
            S_b = S_b + (n_class_samples * a.dot(a.T))
            for sample_index in range(n_class_samples):
                sample = class_samples[:, sample_index]
                a = (sample - mean_class).reshape((-1,1))
                S_w = S_w + a.dot(a.T)
        Epsilon = 0.0001
        eig_val, eig_vec = LA.eigh( np.linalg.inv(S_w + Epsilon*np.identity(S_w.shape[0])).dot(S_b) )
        idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]
        if self.n_components is not None:
            self.U = eig_vec[:, :self.n_components]
            self.eigenvalues = eig_val[:self.n_components]
        else:
            self.U = eig_vec
            self.eigenvalues = eig_val

    def get_U(self):
        return self.U

    def get_eigenvalues(self):
        return self.eigenvalues

    def transform(self, X, Y=None):
        X_transformed = ((self.U).T).dot(X.T)
        X_transformed = X_transformed.T
        return X_transformed

    @staticmethod
    def separate_samples_of_classes(X, y):
        # X --> rows: samples, columns: features
        # return X_separated_classes --> each element of list --> rows: samples, columns: features
        y = np.asarray(y)
        y = y.reshape((-1, 1))
        yX = np.column_stack((y, X))
        yX = yX[yX[:,
                0].argsort()]  # sort array (asscending) with regards to nth column --> https://gist.github.com/stevenvo/e3dad127598842459b68
        y = yX[:, 0]
        X = yX[:, 1:]
        labels_of_classes = list(set(y))
        number_of_classes = len(labels_of_classes)
        dimension_of_data = X.shape[1]
        X_separated_classes = [np.empty((0, dimension_of_data))] * number_of_classes
        class_index = 0
        index_start_new_class = 0
        n_samples = X.shape[0]
        for sample_index in range(1, n_samples):
            if y[sample_index] != y[sample_index - 1]:
                X_separated_classes[class_index] = np.vstack([X_separated_classes[class_index], X[index_start_new_class:sample_index, :]])
                index_start_new_class = sample_index
                class_index = class_index + 1
            if sample_index == n_samples-1:
                X_separated_classes[class_index] = np.vstack([X_separated_classes[class_index], X[index_start_new_class:, :]])
        return X_separated_classes