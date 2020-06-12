import numpy as np
from sklearn.neighbors import NearestNeighbors as KNN  # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


# Sample Reduction with DROP:
# Paper: Reduction Techniques for Instance-Based Learning Algorithms
class SR_Drop:

    def __init__(self, X, Y, n_neighbors):
        # X: rows are features and columns are samples
        # Y: rows are features and columns are samples
        self.X = X
        self.Y = Y
        self.n_dimensions = X.shape[0]
        self.n_samples = X.shape[1]
        self.n_neighbors = n_neighbors

    def sort_samples_withDrop1(self):
        X_sorted = np.zeros(self.X.shape)
        Y_sorted = np.zeros(self.Y.shape)
        last_index_removal_in_X_sorted = self.n_samples - 1
        last_index_removal_in_X = None
        last_index_kept_in_X_sorted = 0
        X_sampleRemoved = self.X
        Y_sampleRemoved = self.Y
        # process every point (whether to remove it or not):
        for sample_index in range(self.n_samples):
            # --- find k-nearest neighbor graph:
            n_neighbors = min(self.n_neighbors, X_sampleRemoved.shape[1])
            connectivity_matrix, connectivity_matrix_having_labels = self.find_KNN_graph(X=X_sampleRemoved, Y=Y_sampleRemoved, n_neighbors=n_neighbors)
            indices_of_neighbors_of_that_sample = [i for i in range(self.n_samples) if connectivity_matrix_having_labels[i, sample_index] != np.nan]
            # --- with the sample:
            classified_correctly_withSample = 0
            for neighbor_sample_index in indices_of_neighbors_of_that_sample:
                label_of_neighbor_sample = Y_sampleRemoved[:, neighbor_sample_index]
                n_similar_samples = np.sum(connectivity_matrix_having_labels[neighbor_sample_index, :] == label_of_neighbor_sample) - 1  # we exclude the sample itself from neighbors
                n_dissimilar_samples = np.sum(connectivity_matrix_having_labels[neighbor_sample_index, :] != label_of_neighbor_sample)
                if n_similar_samples > n_dissimilar_samples:
                    classified_correctly_withSample = classified_correctly_withSample + 1
            # --- without the sample:
            # connectivity_matrix_without_sample = connectivity_matrix_having_labels.copy()
            connectivity_matrix_without_sample = connectivity_matrix_having_labels
            connectivity_matrix_without_sample[:, sample_index] = np.nan
            classified_correctly_withoutSample = 0
            for neighbor_sample_index in indices_of_neighbors_of_that_sample:
                label_of_neighbor_sample = Y_sampleRemoved[:, neighbor_sample_index]
                n_similar_samples = np.sum(connectivity_matrix_without_sample[neighbor_sample_index, :] == label_of_neighbor_sample) - 1  # we exclude the sample itself from neighbors
                n_dissimilar_samples = np.sum(connectivity_matrix_without_sample[neighbor_sample_index, :] != label_of_neighbor_sample)
                if n_similar_samples > n_dissimilar_samples:
                    classified_correctly_withoutSample = classified_correctly_withoutSample + 1
            # --- check whether to remove sample or not:
            if classified_correctly_withoutSample > classified_correctly_withSample:
                # should be removed
                X_sorted[:, last_index_removal_in_X_sorted] = self.X[:, sample_index]
                Y_sorted[:, last_index_removal_in_X_sorted] = self.Y[:, sample_index]
                last_index_removal_in_X_sorted = last_index_removal_in_X_sorted - 1
                last_index_removal_in_X = sample_index
            else:
                # should be kept
                X_sorted[:, last_index_kept_in_X_sorted] = self.X[:, sample_index]
                Y_sorted[:, last_index_kept_in_X_sorted] = self.Y[:, sample_index]
                last_index_kept_in_X_sorted = last_index_kept_in_X_sorted + 1
            # --- remove data:
            if last_index_removal_in_X is not None:
                X_sampleRemoved = np.delete(X_sampleRemoved, last_index_removal_in_X, axis=1)
                Y_sampleRemoved = np.delete(Y_sampleRemoved, last_index_removal_in_X, axis=1)
        return X_sorted, Y_sorted

    def find_KNN_graph(self, X, Y, n_neighbors):
        # input: X, Y --> (rows are features, columns are samples)
        n_samples = X.shape[1]
        knn = KNN(n_neighbors=n_neighbors, algorithm='kd_tree')
        knn.fit(X=X.T)
        connectivity_matrix = knn.kneighbors_graph(X=X.T, n_neighbors=n_neighbors, mode='connectivity')
        connectivity_matrix = connectivity_matrix.toarray()
        # --- replace zeros with nan:
        connectivity_matrix[connectivity_matrix == 0] = np.nan
        # --- replace ones (connectivities) with labels:
        labels = Y.reshape((1, -1))
        repeated_labels_in_rows = np.tile(labels, (n_samples, 1))
        connectivity_matrix_having_labels = np.multiply(connectivity_matrix, repeated_labels_in_rows)
        return connectivity_matrix, connectivity_matrix_having_labels

    def sort_samples_withDrop1_faster(self):
        X_sorted = np.zeros(self.X.shape)
        Y_sorted = np.zeros(self.Y.shape)
        last_index_removal_in_X_sorted = self.n_samples - 1
        last_index_removal_in_X = None
        last_index_kept_in_X_sorted = 0
        X_sampleRemoved = self.X
        Y_sampleRemoved = self.Y
        # --- find k-nearest neighbor graph:
        n_neighbors = min(self.n_neighbors, X_sampleRemoved.shape[1])
        knn = KNN(n_neighbors=n_neighbors, algorithm='kd_tree')
        knn.fit(X=(self.X).T)
        distance_matrix = knn.kneighbors_graph(X=(self.X).T, n_neighbors=n_neighbors, mode='connectivity')
        distance_matrix = distance_matrix.toarray()
        connectivity_matrix = np.zeros((self.n_samples, self.n_samples))
        # process every point (whether to remove it or not):
        for sample_index in range(self.n_samples):
            # --- remove the removed sample from KNN graph:
            if last_index_removal_in_X is not None:
                distance_matrix[:, last_index_removal_in_X] = np.inf  # set to inf so when sorting, it will be last (not neighbor to any point)
            # --- find (updait again) k-nearest neighbors of every sample:
            for sample_index_2 in range(self.n_samples):
                distances_from_neighbors = distance_matrix[sample_index_2, :]
                sorted_neighbors_by_distance = distances_from_neighbors.argsort()
                n_neighbors = min(self.n_neighbors, np.sum(distances_from_neighbors != np.inf))  # in last iterations, the number of left samples becomes less than self.n_neighbors
                for neighbor_index in range(n_neighbors):
                    index_of_neighbor = sorted_neighbors_by_distance[neighbor_index]
                    connectivity_matrix[sample_index_2, index_of_neighbor] = 1
            # --- replace zeros with nan:
            connectivity_matrix[connectivity_matrix == 0] = np.nan
            # --- replace ones (connectivities) with labels:
            labels = self.Y.reshape((1, -1))
            repeated_labels_in_rows = np.tile(labels, (self.n_samples, 1))
            connectivity_matrix_having_labels = np.multiply(connectivity_matrix, repeated_labels_in_rows)
            # --- identifying neighbors of sample (using connectivity matrix):
            indices_of_neighbors_of_that_sample = [i for i in range(self.n_samples) if connectivity_matrix_having_labels[i, sample_index] != np.nan]
            # --- with the sample:
            classified_correctly_withSample = 0
            for neighbor_sample_index in indices_of_neighbors_of_that_sample:
                label_of_neighbor_sample = self.Y[:, neighbor_sample_index]
                n_similar_samples = np.sum(connectivity_matrix_having_labels[neighbor_sample_index, :] == label_of_neighbor_sample) - 1  # we exclude the sample itself from neighbors
                n_dissimilar_samples = np.sum(connectivity_matrix_having_labels[neighbor_sample_index, :] != label_of_neighbor_sample)
                if n_similar_samples > n_dissimilar_samples:
                    classified_correctly_withSample = classified_correctly_withSample + 1
            # --- without the sample:
            # connectivity_matrix_without_sample = connectivity_matrix_having_labels.copy()
            connectivity_matrix_without_sample = connectivity_matrix_having_labels
            connectivity_matrix_without_sample[:, sample_index] = np.nan
            classified_correctly_withoutSample = 0
            for neighbor_sample_index in indices_of_neighbors_of_that_sample:
                label_of_neighbor_sample = self.Y[:, neighbor_sample_index]
                n_similar_samples = np.sum(connectivity_matrix_without_sample[neighbor_sample_index, :] == label_of_neighbor_sample) - 1  # we exclude the sample itself from neighbors
                n_dissimilar_samples = np.sum(connectivity_matrix_without_sample[neighbor_sample_index, :] != label_of_neighbor_sample)
                if n_similar_samples > n_dissimilar_samples:
                    classified_correctly_withoutSample = classified_correctly_withoutSample + 1
            # --- check whether to remove sample or not:
            if classified_correctly_withoutSample > classified_correctly_withSample:
                # should be removed
                X_sorted[:, last_index_removal_in_X_sorted] = self.X[:, sample_index]
                Y_sorted[:, last_index_removal_in_X_sorted] = self.Y[:, sample_index]
                last_index_removal_in_X_sorted = last_index_removal_in_X_sorted - 1
                last_index_removal_in_X = sample_index
            else:
                # should be kept
                X_sorted[:, last_index_kept_in_X_sorted] = self.X[:, sample_index]
                Y_sorted[:, last_index_kept_in_X_sorted] = self.Y[:, sample_index]
                last_index_kept_in_X_sorted = last_index_kept_in_X_sorted + 1
            # --- remove data:
            if last_index_removal_in_X is not None:
                X_sampleRemoved = np.delete(X_sampleRemoved, last_index_removal_in_X, axis=1)
                Y_sampleRemoved = np.delete(Y_sampleRemoved, last_index_removal_in_X, axis=1)
        return X_sorted, Y_sorted
