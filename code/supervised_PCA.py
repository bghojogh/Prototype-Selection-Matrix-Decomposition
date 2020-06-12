import numpy as np
from numpy import linalg as LA

class Supervised_PCA:

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.U = None
        self.eigenvalues = None

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        X_transformed = self.transform(X, Y)
        return X_transformed

    def fit(self, X, Y):
        # X: rows are features and columns are samples
        # Y: rows are features and columns are samples
        n_samples = X.shape[1]
        H = np.eye(n_samples) - ((1/n_samples) * np.ones((n_samples,n_samples)))
        B = (Y.T).dot(Y)
        eig_val, eig_vec = LA.eigh( X.dot(H).dot(B).dot(H).dot(X.T) )
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