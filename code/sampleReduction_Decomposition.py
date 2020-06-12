import numpy as np
import matplotlib.pyplot as plt
import os
from supervised_PCA import Supervised_PCA
from fisher_LDA import Fisher_LDA
from rank_one_downdate import Rank_one_downdate
from sklearn.decomposition import NMF   # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
from sklearn.decomposition import DictionaryLearning   # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html
from sklearn.decomposition import FastICA   # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html#sklearn.decomposition.FastICA
from scipy.linalg import qr as QR  # https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.linalg.qr.html
from scipy.linalg import lu as LU  # https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.linalg.lu.html


# Sample Reduction with Matrix Decomposition (SR_MD):
class SR_MD:

    def __init__(self, X, Y=None):
        # X: rows are features and columns are samples
        # Y: rows are features and columns are samples
        self.X = X
        self.Y = Y
        self.n_dimensions = X.shape[0]
        self.n_samples = X.shape[1]
        self.n_components = None
        X_normalized = [X[:, j] / np.linalg.norm(X[:, j]) for j in range(X.shape[1])]
        X_normalized = np.asarray(X_normalized).T
        # print('...check if X is normalized correctly?')
        # print([np.linalg.norm(X_normalized[:, j]) for j in range(X_normalized.shape[1])])
        self.X_normalized = X_normalized

    def get_n_components(self):
        return self.n_components

    def find_scores_and_sort(self, method):
        # return X_sorted: rows are features, columns are samples
        # return scores_sorted: row vector (each element is score of a sample)
        if method == 'SVD_python':
            U, Lambda, V, k = SR_MD.SVD_python(X=self.X)
            eigenvalues = np.diag(Lambda)
        elif method == 'SVD_Jordan':
            U, Lambda, V, k = SR_MD.SVD_Jordan(X=self.X)
            eigenvalues = np.diag(Lambda)
        elif method == 'SPCA':
            U, eigenvalues, k = SR_MD.SPCA(X=self.X, Y=self.Y)
        elif method == 'FLDA':
            U, eigenvalues, k = SR_MD.FLDA(X=self.X, Y=self.Y)
        elif method == 'R1D':
            U, k = SR_MD.R1D(X=self.X)
            U = self.__normalize_U(U)
            eigenvalues = None
        elif method == 'modified_R1D':
            scores, k = SR_MD.modofied_R1D(X=self.X)
        elif method == 'NMF_sklearn':
            U, k = SR_MD.NMF_sklearn(X=self.X)
            U = self.__normalize_U(U)
            eigenvalues = None
        elif method == 'Dictionary_learning':
            U, k = SR_MD.Dictionary_learning(X=self.X)
            U = self.__normalize_U(U)
            eigenvalues = None
        elif method == 'ICA':
            U, k = SR_MD.ICA(X=self.X)
            U = self.__normalize_U(U)
            eigenvalues = None
        elif method == 'QR_decomposition':
            U, k = SR_MD.QR_decomposition(X=self.X)
            eigenvalues = None
        elif method == 'LU_decomposition':
            U, k = SR_MD.LU_decomposition(X=self.X)
            U = self.__normalize_U(U)
            eigenvalues = None
        elif method == 'sorted_by_distance_from_mean':
            k = None
            scores = SR_MD.sorted_by_distance_from_mean(X=self.X, Y=self.Y)
        elif method == 'stratified_sampling':
            k = None
            scores = SR_MD.stratified_sampling(X=self.X)
        if method != 'modified_R1D' and method != 'sorted_by_distance_from_mean' and method != 'stratified_sampling':
            # print('...check if U is normal?')
            # print([np.linalg.norm(U[:, j]) for j in range(U.shape[1])])
            scores = self.__calculate_scores(U=U, eigenvalues=eigenvalues)
        X_sorted, Y_sorted, scores_sorted = self._sort_samples(scores=scores)
        self.n_components = k
        return X_sorted, Y_sorted, scores_sorted, scores

    def __calculate_scores(self, U, eigenvalues):
        # return: scores: a row vector (each element is score of a sample)
        k = U.shape[1]
        if eigenvalues is not None:
            if eigenvalues[k-1] < 0:
                eigenvalues = eigenvalues - 2*eigenvalues[k-1]
            weights = np.asarray([(1 / eigenvalues[i]) / np.sum((1 / eigenvalues)) for i in range(k)]).reshape((-1, 1))
            # weights = np.asarray([np.exp(-1*eigenvalues[i]) / np.sum(np.exp(-1*eigenvalues)) for i in range(k)]).reshape((-1, 1))
            # weights = np.asarray([eigenvalues[k - i - 1] / np.sum(eigenvalues) for i in range(k)]).reshape((-1, 1))
        else:
            weights = np.asarray([(2 * (i + 1)) / (k * (k + 1)) for i in range(k)]).reshape((-1, 1))
        # weights = np.asarray([(2 * (i + 1)) / (k * (k + 1)) for i in range(k)]).reshape((-1, 1))
        abs_matrix = np.absolute(((self.X_normalized).T).dot(U))
        epsilon = 0.001
        abs_matrix[np.where(abs_matrix < epsilon)] = epsilon
        scores = (-1 * np.log(abs_matrix)).dot(weights)
        # S = np.absolute(((self.X_normalized).T).dot(U))
        # scores = np.ones((self.n_samples, 1))
        # for j in range(S.shape[1]):
        #     scores = np.multiply(scores, (S[:, j]).reshape((-1, 1)))
        scores = scores.T
        return scores

    def _sort_samples(self, scores):
        # output: X_sorted, Y_sorted, scores_sorted --> (rows are features, columns are samples)
        if self.Y is None:
            X_with_scores = np.vstack((scores, self.X))
            # sort matrix with respect to values in first row:
            X_with_scores_sorted = SR_MD.sort_matrix(X=X_with_scores, withRespectTo_columnOrRow='row', index_columnOrRow=0, descending=True)
            X_sorted = X_with_scores_sorted[1:, :]
            scores_sorted = X_with_scores_sorted[0, :]
            Y_sorted = None
        else:
            X_with_scores = np.vstack((scores, self.X))
            X_with_scores_and_Y = np.vstack((X_with_scores, self.Y))
            # sort matrix with respect to values in first row:
            X_with_scores_and_Y_sorted = SR_MD.sort_matrix(X=X_with_scores_and_Y, withRespectTo_columnOrRow='row', index_columnOrRow=0, descending=True)
            X_sorted = X_with_scores_and_Y_sorted[1:1 + self.n_dimensions, :]
            Y_sorted = X_with_scores_and_Y_sorted[self.n_dimensions + 1:, :]
            scores_sorted = X_with_scores_and_Y_sorted[0, :]
        return X_sorted, Y_sorted, scores_sorted

    def __normalize_U(self, U):
        # normalize U:
        # U_normalized = [U[:, j] / np.linalg.norm(U[:, j]) for j in range(U.shape[1])]
        # U = np.asarray(U_normalized).T
        for j in range(U.shape[1]):
            if np.linalg.norm(U[:, j]) != 0:
                U[:, j] = U[:, j] / np.linalg.norm(U[:, j])
        return U

    @staticmethod
    def LU_decomposition(X):
        # input: X (rows are features, columns are samples)
        # Outputs: U (columns are projection directions)
        n_dimensions = X.shape[0]
        n_samples = X.shape[1]
        k = min(n_dimensions, n_samples)
        P, Lower, Upper = LU(a=X)
        U = P.dot(Lower)
        # print(X.shape, U.shape)
        return U, k

    @staticmethod
    def QR_decomposition(X):
        # input: X (rows are features, columns are samples)
        # Outputs: U (columns are projection directions)
        n_dimensions = X.shape[0]
        n_samples = X.shape[1]
        k = min(n_dimensions, n_samples)
        Q, R = QR(a=X, mode='economic')  # mode=economic sets min(n_dimensions, n_samples)
        U = Q
        # print(X.shape, U.shape)
        return U, k

    @staticmethod
    def NMF_sklearn(X):
        # input: X (rows are features, columns are samples)
        # Outputs: U (columns are projection directions) --> note: X = UV
        n_dimensions = X.shape[0]
        n_samples = X.shape[1]
        k = min(n_dimensions, n_samples)
        model = NMF(n_components=k, init='random', random_state=0, tol=1e-20, max_iter=int(1e4))
        # model = NMF(n_components=k, init='random', random_state=0, tol=1e-20, max_iter=int(1e3))
        W = model.fit_transform(X)
        H = model.components_
        U = W
        V = H
        # print(np.linalg.norm(X - W.dot(H)))
        return U, k

    @staticmethod
    def Dictionary_learning(X):
        # input: X (rows are features, columns are samples)
        # Outputs: U (columns are projection directions) --> note: X = UV
        n_dimensions = X.shape[0]
        n_samples = X.shape[1]
        k = min(n_dimensions, n_samples)
        # model = DictionaryLearning(n_components=k, tol=1e-20, max_iter=int(1e4))
        model = DictionaryLearning(n_components=k)
        model.fit(X)
        V = model.components_
        U = X.dot(V.T).dot(np.linalg.inv(V.dot(V.T)))  # X=UV --> X:d*n, U:d*k, V:k*n --> XV' = UVV' --> U = XV'(VV')^{-1}
        # print(X.shape, U.shape, V.shape, k)
        return U, k

    @staticmethod
    def ICA(X):
        # input: X (rows are features, columns are samples)
        # Outputs: U ---> C (covariance matrix): d*d, C = WW' + covariance_of_noise (?)
        # Outputs: U ---> C (covariance matrix): d*d, C = HW + ... (?) --> W (mixing matrix): d*k, H (unmixing matrix, independent components matrix): k*d
        n_dimensions = X.shape[0]
        n_samples = X.shape[1]
        k = min(n_dimensions, n_samples)
        # model = FastICA(n_components=k, tol=1e-20, max_iter=int(1e4))
        model = FastICA(n_components=k)
        X = X.T  # --> rows become samples and columns become features
        model.fit(X)
        W = model.components_   # (unmixing or components matrix): k*d where X.shape[0]=n, X.shape[1]=d
        H = model.mixing_       # (mixing matrix): d*k where X.shape[0]=n, X.shape[1]=d
        U = W.T
        # print(X.shape, W.shape, H.shape, U.shape, k)
        return U, k

    @staticmethod
    def modofied_R1D(X):
        # input: X (rows are features, columns are samples)
        # Outputs: U (columns are projection directions)
        n_dimensions = X.shape[0]
        n_samples = X.shape[1]
        rank_one_downdate = Rank_one_downdate()
        X_normalized_std = np.asarray([X[:, j] / np.std(X[:, j]) for j in range(X.shape[1])]).T
        clusters_of_N, k = rank_one_downdate.modified_R1_decompose(X=X_normalized_std)
        # clusters_of_N, k = rank_one_downdate.modified_R1_decompose(X=X)
        total_mean = X.mean(axis=1)
        n_clusters = len(clusters_of_N)
        print('# extracted clusters: ', n_clusters)
        mean_of_cluster = [None] * n_clusters
        X_m = np.zeros((n_dimensions, n_samples))  # matrix of means of clusters corresponding to samples
        for cluster_index in range(n_clusters):
            cluster_samples_indices = clusters_of_N[cluster_index]
            cluster_samples = np.asarray([X[:, j] for j in cluster_samples_indices]).T
            mean_of_cluster[cluster_index] = cluster_samples.mean(axis=1)
            X_m[:, cluster_samples_indices] = mean_of_cluster[cluster_index].reshape((-1, 1))
        # normalize vectors:
        X_normalized = np.asarray([X[:, j] / np.linalg.norm(X[:, j]) for j in range(X.shape[1])]).T
        X_m_normalized = np.asarray([X_m[:, j] / np.linalg.norm(X_m[:, j]) for j in range(X_m.shape[1])]).T
        total_mean_normalized = total_mean / np.linalg.norm(total_mean)
        # calculate scores:
        score_within_clusters = np.diagonal((X_normalized.T).dot(X_m_normalized))
        score_clusters = (X_m_normalized.T).dot(total_mean_normalized)
        # scale scores to range [0, 1]:
        score_within_clusters = 0.5 * (score_within_clusters + 1)
        score_clusters = 0.5 * (score_clusters + 1)
        scores = np.multiply(score_within_clusters, score_clusters)
        return scores, k

    @staticmethod
    def R1D(X):
        # input: X (rows are features, columns are samples)
        # Outputs: U (columns are projection directions)
        n_dimensions = X.shape[0]
        n_samples = X.shape[1]
        rank_one_downdate = Rank_one_downdate()
        U, V, k = rank_one_downdate.R1D_decompose(X=X)
        # print('//////////////////////')
        # print(U.shape, V.shape)
        # print(np.linalg.norm(U.dot(V.T) - X))
        return U, k

    @staticmethod
    def sorted_by_distance_from_mean(X, Y):
        # input: X (rows are features, columns are samples)
        # input: Y (rows are labels, columns are samples)
        # Outputs: scores (a row vector)
        n_dimensions = X.shape[0]
        n_samples = X.shape[1]
        if Y is not None:
            X_separated_classes = SR_MD.separate_samples_of_classes(X=X.T, y=Y.T)
            labels_of_classes = sorted(set(Y.ravel().tolist()))
            n_classes = len(X_separated_classes)
            class_mean = [None] * n_classes
            for class_index in range(n_classes):
                X_class = X_separated_classes[class_index].T
                class_mean[class_index] = X_class.mean(axis=1)
            matrix_of_means = np.zeros((n_dimensions, n_samples))
            for sample_index in range(n_samples):
                label_of_sample = int(Y[:, sample_index])
                class_index = [i for i, x in enumerate(labels_of_classes) if x == label_of_sample]  # https://stackoverflow.com/questions/364621/how-to-get-items-position-in-a-list
                class_index = class_index[0]  # to convert [number] to number
                matrix_of_means[:, sample_index] = class_mean[class_index]
        else:
            mean = X.mean(axis=1)
            matrix_of_means = np.zeros((n_dimensions, n_samples))
            for sample_index in range(n_samples):
                matrix_of_means[:, sample_index] = mean
        distances_of_samples_from_class_means = np.linalg.norm(X - matrix_of_means, axis=0)
        scores = np.reciprocal(distances_of_samples_from_class_means)
        return scores

    @staticmethod
    def stratified_sampling(X):
        # input: X (rows are features, columns are samples)
        # Outputs: scores (a row vector)
        n_samples = X.shape[1]
        # scores = np.random.rand(1, n_samples)
        scores = np.random.uniform(0, 1, n_samples)
        return scores

    @staticmethod
    def SPCA(X, Y):
        # input: X (rows are features, columns are samples)
        # input: Y (rows are labels, columns are samples)
        # Outputs: U (columns are projection directions)
        n_dimensions = X.shape[0]
        n_samples = X.shape[1]
        k = min(n_dimensions, n_samples)
        supervised_PCA = Supervised_PCA(n_components=k)
        supervised_PCA.fit(X=X, Y=Y)
        U = supervised_PCA.get_U()
        eigenvalues = supervised_PCA.get_eigenvalues()
        return U, eigenvalues, k

    @staticmethod
    def FLDA(X, Y):
        # input: X (rows are features, columns are samples)
        # input: Y (rows are labels, columns are samples)
        # Outputs: U (columns are projection directions)
        n_dimensions = X.shape[0]
        n_samples = X.shape[1]
        k = min(n_dimensions, n_samples)
        fisher_LDA = Fisher_LDA(n_components=k)
        fisher_LDA.fit(X=X, Y=Y)
        U = fisher_LDA.get_U()
        eigenvalues = fisher_LDA.get_eigenvalues()
        return U, eigenvalues, k

    @staticmethod
    def SVD_python(X):
        # input: X
        # Outputs: U, Lambda, V (X = U Lambda V.T)
        # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.linalg.svd.html
        # https://stackoverflow.com/questions/24913232/using-numpy-np-linalg-svd-for-singular-value-decomposition
        U, Lambda_vector, V_transpose = np.linalg.svd(X, full_matrices=False)
        n_dimensions = X.shape[0]
        n_samples = X.shape[1]
        k = min(n_dimensions, n_samples)
        return U, np.diag(Lambda_vector), V_transpose.T, k

    @staticmethod
    def SVD_Jordan(X):
        # input: X
        # Outputs: U, Lambda, V (X = U Lambda V.T)
        n_dimensions = X.shape[0]
        n_samples = X.shape[1]
        k = min(X.shape[0], X.shape[1])
        U = np.zeros((n_dimensions, k))
        Lambda = np.zeros((k, k))
        V = np.zeros((n_samples, k))
        for j in range(k):
            u_bar = np.random.rand(n_dimensions, 1) + 1  # in range [1, 2)
            sigma = np.linalg.norm(u_bar)
            u = u_bar / sigma
            v = np.zeros((n_samples, 1))
            sigma = 0
            for counter in range(10**8):
                u_previous_iteration = u
                v_previous_iteration = v
                sigma_previous_iteration = sigma
                v_bar = (X.T).dot(u)
                v = v_bar / np.linalg.norm(v_bar)
                u_bar = X.dot(v)
                sigma = np.linalg.norm(u_bar)
                u = u_bar / sigma
                if np.linalg.norm(u - u_previous_iteration) < 1e-4 and \
                   np.linalg.norm(v - v_previous_iteration) < 1e-4 and \
                   np.linalg.norm(sigma - sigma_previous_iteration) < 1e-4:
                        break
            X = X - u.dot(sigma).dot(v.T)
            U[:, j] = u.ravel()
            V[:, j] = v.ravel()
            Lambda[j, j] = sigma
        return U, Lambda, V, k

    @staticmethod
    def twoD_plot_samples_unsupervised(X, color_of_plot, axes_labels='None', marker_size=50, save_figures=True, path_save='./', name_save='image', show_images=False):
        # X: rows are features and columns are samples
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plt1 = plt.scatter(X[0, :], X[1, :], c=color_of_plot, marker='o', s=marker_size, edgecolors='k')
        plt.xlabel(axes_labels[0])
        plt.ylabel(axes_labels[1])
        if save_figures:
            name_of_image = name_save
            format_of_save = 'png'
            if not os.path.exists(os.path.dirname(path_save)):
                os.makedirs(os.path.dirname(path_save))
            plt.savefig(path_save + name_of_image + '.' + format_of_save, dpi=300)  # if don't want borders: bbox_inches='tight'
        if show_images: plt.show()

    @staticmethod
    def twoD_plot_sorted_samples_unsupervised(X_sorted, color_of_plot, axes_labels='None', biggest_marker_size=500,
                                              save_figures=True, path_save='./', name_save='image', show_images=False):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        n_samples = X_sorted.shape[1]
        smallest_marker_size = 1
        step_marker_size = (biggest_marker_size - smallest_marker_size) / n_samples
        for sample_index in range(n_samples):
            sample = X_sorted[:, sample_index]
            rank = sample_index
            marker_size = biggest_marker_size - (step_marker_size * rank)
            plt1 = plt.scatter(sample[0], sample[1], c=color_of_plot, marker='o', s=marker_size, edgecolors='k')
        plt.xlabel(axes_labels[0])
        plt.ylabel(axes_labels[1])
        if save_figures:
            name_of_image = name_save
            format_of_save = 'png'
            if not os.path.exists(os.path.dirname(path_save)):
                os.makedirs(os.path.dirname(path_save))
            plt.savefig(path_save + name_of_image + '.' + format_of_save, dpi=300)  # if don't want borders: bbox_inches='tight'
        if show_images: plt.show()

    @staticmethod
    def twoD_plot_samples_classification(X_sorted, Y_sorted, color_of_plot, markers, axes_labels='None', show_legends=True,
                                         marker_size=50, save_figures=True, path_save='./', name_save='image', show_images=False):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        labels = list(set(Y_sorted.ravel()))  # set() removes redundant numbers in y and sorts them
        n_classes = len(labels)
        X_separated_classes = SR_MD.separate_samples_of_classes(X=X_sorted.T, y=Y_sorted.T)
        for class_index in range(n_classes):
            class_samples = X_separated_classes[class_index].T
            n_samples = class_samples.shape[1]
            smallest_marker_size = 1
            for sample_index in range(n_samples):
                sample = class_samples[:, sample_index]
                rank = sample_index
                plt1 = plt.scatter(sample[0], sample[1], c=color_of_plot[class_index], marker=markers[class_index],
                                   s=marker_size, edgecolors='k')
                if rank == n_samples - 10:  # a small marker
                    plt_handler = plt1
            plt_handler.set_label('Class ' + str(class_index))
        if show_legends:
            ax.legend()  # https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html     # https://stackoverflow.com/questions/40351026/plotting-a-simple-3d-numpy-array-using-matplotlib
        plt.xlabel(axes_labels[0])
        plt.ylabel(axes_labels[1])
        if save_figures:
            name_of_image = name_save
            format_of_save = 'png'
            if not os.path.exists(os.path.dirname(path_save)):
                os.makedirs(os.path.dirname(path_save))
            plt.savefig(path_save + name_of_image + '.' + format_of_save,
                        dpi=300)  # if don't want borders: bbox_inches='tight'
        if show_images: plt.show()

    @staticmethod
    def twoD_plot_sorted_samples_classification(X_sorted, Y_sorted, color_of_plot, markers, axes_labels='None', show_legends=True,
                                                biggest_marker_size=500, save_figures=True, path_save='./', name_save='image', show_images=False):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        labels = list(set(Y_sorted.ravel()))  # set() removes redundant numbers in y and sorts them
        n_classes = len(labels)
        X_separated_classes = SR_MD.separate_samples_of_classes(X=X_sorted.T, y=Y_sorted.T)
        for class_index in range(n_classes):
            class_samples = X_separated_classes[class_index].T
            n_samples = class_samples.shape[1]
            smallest_marker_size = 1
            step_marker_size = (biggest_marker_size - smallest_marker_size) / n_samples
            for sample_index in range(n_samples):
                sample = class_samples[:, sample_index]
                rank = sample_index
                marker_size = biggest_marker_size - (step_marker_size * rank)
                plt1 = plt.scatter(sample[0], sample[1], c=color_of_plot[class_index], marker=markers[class_index], s=marker_size, edgecolors='k')
                if rank == n_samples - 10:  # a small marker
                    plt_handler = plt1
            plt_handler.set_label('Class ' + str(class_index))
        if show_legends:
            ax.legend()  # https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html     # https://stackoverflow.com/questions/40351026/plotting-a-simple-3d-numpy-array-using-matplotlib
        plt.xlabel(axes_labels[0])
        plt.ylabel(axes_labels[1])
        if save_figures:
            name_of_image = name_save
            format_of_save = 'png'
            if not os.path.exists(os.path.dirname(path_save)):
                os.makedirs(os.path.dirname(path_save))
            plt.savefig(path_save + name_of_image + '.' + format_of_save, dpi=300)  # if don't want borders: bbox_inches='tight'
        if show_images: plt.show()

    @staticmethod
    def sort_matrix(X, withRespectTo_columnOrRow='column', index_columnOrRow=0, descending=True):
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

    @staticmethod
    def separate_samples_of_classes(X, y):  # it does not change the order of the samples within every class
        # X --> rows: samples, columns: features
        # return X_separated_classes --> each element of list --> rows: samples, columns: features
        y = np.asarray(y)
        y = y.reshape((-1, 1)).ravel()
        labels_of_classes = sorted(set(y.ravel().tolist()))
        n_samples = X.shape[0]
        n_dimensions = X.shape[1]
        n_classes = len(labels_of_classes)
        X_separated_classes = [np.empty((0, n_dimensions))] * n_classes
        for class_index in range(n_classes):
            X_separated_classes[class_index] = X[y == labels_of_classes[class_index], :]
        return X_separated_classes

    # @staticmethod
    # def separate_samples_of_classes(X, y):
    #     # X --> rows: samples, columns: features
    #     # return X_separated_classes --> each element of list --> rows: samples, columns: features
    #     y = np.asarray(y)
    #     y = y.reshape((-1, 1))
    #     yX = np.column_stack((y, X))
    #     yX = yX[yX[:,
    #             0].argsort()]  # sort array (asscending) with regards to nth column --> https://gist.github.com/stevenvo/e3dad127598842459b68
    #     y = yX[:, 0]
    #     X = yX[:, 1:]
    #     labels_of_classes = list(set(y))
    #     number_of_classes = len(labels_of_classes)
    #     dimension_of_data = X.shape[1]
    #     X_separated_classes = [np.empty((0, dimension_of_data))] * number_of_classes
    #     class_index = 0
    #     index_start_new_class = 0
    #     n_samples = X.shape[0]
    #     for sample_index in range(1, n_samples):
    #         if y[sample_index] != y[sample_index - 1]:
    #             X_separated_classes[class_index] = np.vstack(
    #                 [X_separated_classes[class_index], X[index_start_new_class:sample_index, :]])
    #             index_start_new_class = sample_index
    #             class_index = class_index + 1
    #         if sample_index == n_samples - 1:
    #             X_separated_classes[class_index] = np.vstack(
    #                 [X_separated_classes[class_index], X[index_start_new_class:, :]])
    #     return X_separated_classes