import numpy as np
from sklearn.neighbors import NearestNeighbors as KNN  # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


# Sample Reduction with edited Nearest Neighbor (ENN):
# Paper: Asymptotic Properties of Nearest Neighbor Rules Using Edited Data
class SR_edited_NN:

    def __init__(self, X, Y, n_neighbors):
        # X: rows are features and columns are samples
        # Y: rows are features and columns are samples
        self.X = X
        self.Y = Y
        self.n_dimensions = X.shape[0]
        self.n_samples = X.shape[1]
        self.n_neighbors = n_neighbors

    def sort_samples_withENN(self):
        # --- find k-nearest neighbor graph:
        n_neighbors = self.n_neighbors
        knn = KNN(n_neighbors=n_neighbors, algorithm='kd_tree')
        knn.fit(X=(self.X).T)
        connectivity_matrix = knn.kneighbors_graph(X=(self.X).T, n_neighbors=n_neighbors, mode='connectivity')
        connectivity_matrix = connectivity_matrix.toarray()
        # --- replace zeros with nan:
        connectivity_matrix[connectivity_matrix == 0] = np.nan
        # --- replace ones (connectivities) with labels:
        labels = (self.Y).reshape((1, -1))
        repeated_labels_in_rows = np.tile(labels, (self.n_samples, 1))
        connectivity_matrix_having_labels = np.multiply(connectivity_matrix, repeated_labels_in_rows)
        # --- find scores of samples:
        scores = np.zeros((1, self.n_samples))
        for sample_index in range(self.n_samples):
            label_of_sample = self.Y[:, sample_index]
            score_of_sample = np.sum(connectivity_matrix_having_labels[sample_index, :] == label_of_sample) - 1  # we exclude the sample itself from neighbors
            scores[:, sample_index] = score_of_sample
        # --- find scores of samples:
        X_sorted, Y_sorted, scores_sorted = self.sort_samples(scores=scores, X=self.X, Y=self.Y)
        return X_sorted, Y_sorted, scores_sorted


    def sort_samples(self, scores, X, Y):
        # input: X, Y --> (rows are features, columns are samples)
        # output: X_sorted, Y_sorted, scores_sorted --> (rows are features, columns are samples)
        n_dimensions = X.shape[0]
        n_samples = X.shape[1]
        if Y is None:
            X_with_scores = np.vstack((scores, X))
            # sort matrix with respect to values in first row:
            X_with_scores_sorted = self.sort_matrix(X=X_with_scores, withRespectTo_columnOrRow='row', index_columnOrRow=0, descending=True)
            X_sorted = X_with_scores_sorted[1:, :]
            scores_sorted = X_with_scores_sorted[0, :]
            Y_sorted = None
        else:
            X_with_scores = np.vstack((scores, X))
            X_with_scores_and_Y = np.vstack((X_with_scores, Y))
            # sort matrix with respect to values in first row:
            X_with_scores_and_Y_sorted = self.sort_matrix(X=X_with_scores_and_Y, withRespectTo_columnOrRow='row', index_columnOrRow=0, descending=True)
            X_sorted = X_with_scores_and_Y_sorted[1:1 + n_dimensions, :]
            Y_sorted = X_with_scores_and_Y_sorted[n_dimensions + 1:, :]
            scores_sorted = X_with_scores_and_Y_sorted[0, :]
        return X_sorted, Y_sorted, scores_sorted

    def sort_matrix(self, X, withRespectTo_columnOrRow='column', index_columnOrRow=0, descending=True):
        # I googled: python sort matrix by first row --> https://gist.github.com/stevenvo/e3dad127598842459b68
        # https://stackoverflow.com/questions/37856290/python-argsort-in-descending-order-for-2d-array
        # sort array with regards to nth column or row:
        if withRespectTo_columnOrRow == 'column':
            if descending is True:
                X = X[X[:, index_columnOrRow].argsort()][::-1]
            else:
                X = X[X[:, index_columnOrRow].argsort()]
        elif withRespectTo_columnOrRow == 'row':
            X = X.T
            if descending is True:
                X = X[X[:, index_columnOrRow].argsort()][::-1]
            else:
                X = X[X[:, index_columnOrRow].argsort()]
            X = X.T
        return X