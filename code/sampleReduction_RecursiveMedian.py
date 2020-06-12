import numpy as np


# Sample Reduction with Median:
# Paper: On-demand data numerosity reduction for learning artifacts
class SR_RecursiveMedian:

    def __init__(self, X, Y=None, for_classification=True):
        # X: rows are features and columns are samples
        # Y: rows are features and columns are samples
        self.X = X
        self.Y = Y
        self.n_dimensions = X.shape[0]
        self.n_samples_ofClass = None
        self.distances_sorted = None
        self.indices_of_already_selected_samples = None
        self.IDs_of_already_selected_samples = None
        self.for_classification = for_classification

    def sort_samples_with_RecursiveMedian(self):
        if self.for_classification is True:
            X_separated_classes = self.separate_samples_of_classes(X=(self.X).T, y=(self.Y).ravel())
            n_classes = len(X_separated_classes)
            X_final_sorted = np.empty((self.n_dimensions, 0))
            Y_final_sorted = np.empty(((self.Y).shape[0], 0))
            for class_index in range(n_classes):
                samples_of_class = X_separated_classes[class_index].T
                n_samples_of_class = samples_of_class.shape[1]
                self.n_samples_ofClass = n_samples_of_class
                labels_of_samples_of_class = np.ones((1, n_samples_of_class)) * class_index
                X_final_sorted_ofClass, Y_final_sorted_ofClass = self.__a_piece_of_code(X=samples_of_class, Y=labels_of_samples_of_class)
                X_final_sorted = np.column_stack((X_final_sorted, X_final_sorted_ofClass))
                Y_final_sorted = np.column_stack((Y_final_sorted, Y_final_sorted_ofClass))
        else:
            self.n_samples_ofClass = (self.X).shape[1]
            X_final_sorted, Y_final_sorted = self.__a_piece_of_code(X=self.X, Y=self.Y)
        return X_final_sorted, Y_final_sorted

    def __a_piece_of_code(self, X, Y):
        X_sorted_by_distance, Y_sorted_by_distance, distances_sorted = self.sort_by_distance_from_mean(X=X, Y=Y)
        self.distances_sorted = distances_sorted
        self.indices_of_already_selected_samples = []
        self.IDs_of_already_selected_samples = []
        self.sort_samples_recursively(index_start=0 + 1, index_end=self.n_samples_ofClass - 1 - 1, root_ID='')
        IDs_of_selected_samples = np.asarray([int(x) for x in self.IDs_of_already_selected_samples])
        indices_of_selected_samples = np.asarray(self.indices_of_already_selected_samples)
        IDs_and_indices_of_selected_samples = np.vstack((IDs_of_selected_samples, indices_of_selected_samples))
        # sort with respect to IDs in ascending order: 1, 10, 11, 100, 101, 110, 111, 1000, 1001, ...
        # note: first root had ID: 1, left children had ID: rootID + '0', right children had ID: rootID + '1'
        sorted_IDs_and_indices = self.sort_matrix(X=IDs_and_indices_of_selected_samples, withRespectTo_columnOrRow='row', index_columnOrRow=0, descending=False)
        X_final_sorted = np.zeros(X.shape)
        if self.Y is not None:
            Y_final_sorted = np.zeros(Y.shape)
        else:
            Y_final_sorted = None
        X_final_sorted[:, 0] = X_sorted_by_distance[:, 0]
        X_final_sorted[:, 1] = X_sorted_by_distance[:, -1]
        if self.Y is not None:
            Y_final_sorted[:, 0] = Y_sorted_by_distance[:, 0]
            Y_final_sorted[:, 1] = Y_sorted_by_distance[:, -1]
        counter = 2
        for i in range(sorted_IDs_and_indices.shape[1]):
            sample_index = sorted_IDs_and_indices[1, i]
            X_final_sorted[:, counter] = X_sorted_by_distance[:, sample_index]
            if self.Y is not None:
                Y_final_sorted[:, counter] = Y_sorted_by_distance[:, sample_index]
            counter = counter + 1
        return X_final_sorted, Y_final_sorted

    def sort_samples_recursively(self, index_start, index_end, root_ID):
        # -------- determining start and end indices and check termination of recursive calls:
        if index_start is not None and index_end is not None:  # means it is level 1 (whole samples)
            region = 'whole'
            ID = '1'
        elif index_start is None and index_end is not None:  # means it is a left region
            region = 'left'
            ID = root_ID + '0'
            temp = np.asarray(self.indices_of_already_selected_samples)
            temp = temp[temp <= index_end]  # find the selected indices at left of index_end
            if len(temp) == 0:  # we have no selected sample at left
                if index_end == 1:  # region has only one sample (the second sorted sample) --> so take that sample
                    index_of_selected_sample = index_end
                    self.indices_of_already_selected_samples.extend([index_of_selected_sample])
                    self.IDs_of_already_selected_samples.append(ID)
                    return
                else:
                    index_start = 0 + 1  # we don't take the first sample because it's always selected as first best sample
            elif np.max(temp) == index_end:  # if the sample was already selected itself
                return
            elif np.max(temp) == index_end - 1:  # region has only one sample --> so take that sample
                index_of_selected_sample = index_end
                self.indices_of_already_selected_samples.extend([index_of_selected_sample])
                self.IDs_of_already_selected_samples.append(ID)
                return
            else:
                index_start = np.max(temp) + 1  # go until the largest already selected sample in left region
        elif index_start is not None and index_end is None:  # means it is a right region
            region = 'right'
            ID = root_ID + '1'
            temp = np.asarray(self.indices_of_already_selected_samples)
            temp = temp[temp >= index_start]  # find the selected indices at right of index_end
            if len(temp) == 0:  # we have no selected sample at right
                if index_start == (self.n_samples_ofClass - 1) - 1:  # region has only one sample (the second last sorted sample) --> so take that sample
                    index_of_selected_sample = index_start
                    self.indices_of_already_selected_samples.extend([index_of_selected_sample])
                    self.IDs_of_already_selected_samples.append(ID)
                    return
                else:
                    index_end = (self.n_samples_ofClass - 1) - 1  # we don't take the last sample because it's always selected as second best sample
            elif np.min(temp) == index_start:  # if the sample was already selected itself
                return
            elif np.min(temp) == index_start + 1:  # region has only one sample --> so take that sample
                index_of_selected_sample = index_start
                self.indices_of_already_selected_samples.extend([index_of_selected_sample])
                self.IDs_of_already_selected_samples.append(ID)
                return
            else:
                index_end = np.min(temp) - 1   # go until the smallest already selected sample in right region
        # -------- recursive function calls:
        E_measure_max = -1e5
        for sample_index in range(index_start, index_end+1):
            # sample = self.X_sorted_by_distance[:, sample_index]
            cardinality_of_left = sample_index - index_start  # cardinality of closer samples to mean (left samples) in region
            cardinality_of_right = index_end - sample_index  # cardinality of farther samples to mean (right samples) in region
            d_xPlus1 = self.distances_sorted[sample_index + 1]
            d_x = self.distances_sorted[sample_index]
            # print(cardinality_of_left, cardinality_of_right, index_start, index_end, sample_index, region, ID)
            E_measure = (d_xPlus1 - d_x) * (min(cardinality_of_left, cardinality_of_right) / max(cardinality_of_left, cardinality_of_right))
            if E_measure > E_measure_max:
                E_measure_max = E_measure
                # selected_sample = sample
                index_of_selected_sample = sample_index
        # -------- save the index and ID of selected sample:
        self.indices_of_already_selected_samples.extend([index_of_selected_sample])
        self.IDs_of_already_selected_samples.append(ID)
        # -------- recursive function calls:
        if index_of_selected_sample == self.n_samples_ofClass - 1 - 1:  # we don't take the last sample because it's always selected as second best sample
            # only go to left region:
            self.sort_samples_recursively(index_start=None, index_end=index_of_selected_sample - 1, root_ID=ID)
        elif index_of_selected_sample == 0 + 1:  # we don't take the first sample because it's always selected as first best sample
            # only go to left region:
            self.sort_samples_recursively(index_start=index_of_selected_sample + 1, index_end=None, root_ID=ID)
        else:
            # go to left region:
            self.sort_samples_recursively(index_start=None, index_end=index_of_selected_sample - 1, root_ID=ID)
            # go to right region:
            self.sort_samples_recursively(index_start=index_of_selected_sample + 1, index_end=None, root_ID=ID)

    def sort_by_distance_from_mean(self, X, Y):
        # input: X (rows are features, columns are samples)
        # input: Y (rows are labels, columns are samples)
        # Outputs: scores (a row vector)
        n_dimensions = X.shape[0]
        n_samples = X.shape[1]
        # if Y is not None:
        if self.for_classification is True:
            X_separated_classes = self.separate_samples_of_classes(X=X.T, y=Y.T)
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
        X_sorted, Y_sorted, scores_sorted = self.sort_samples(scores=scores, X=X, Y=Y)
        distances_sorted = np.reciprocal(scores_sorted)  # because scores where reciprocal of distances
        return X_sorted, Y_sorted, distances_sorted

    def separate_samples_of_classes(self, X, y):  # it does not change the order of the samples within every class
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