import numpy as np
from numpy import linalg as LA

class Rank_one_downdate:

    def __init__(self, gamma_bar=None):
        self.W = None
        self.H = None
        if gamma_bar is not None:
            self.gamma_bar = gamma_bar
        else:
            self.gamma_bar = None
        self.all_rows_in_submatrix = None

    def R1D_decompose(self, X):
        # return W (n_rows*k) and H (n_columns*k) --> X = W * H.T
        self.all_rows_in_submatrix = False
        self.gamma_bar = 100   # 100
        n_iterations = 10 ** 3   # 10**3
        termination_criterion = 1
        n_rows = X.shape[0]
        n_columns = X.shape[1]
        k = min(n_rows, n_columns)
        self.W = np.zeros((n_rows, k))
        self.H = np.zeros((n_columns, k))
        X_copy = np.copy(X)
        index_truncate_W_and_H = None
        for j in range(k):
            print('...j: ', j, ' out of ', k)
            D, N, u, v, sigma = self.extract_approximately_rankOne_submatrix(X=X_copy, n_iterations=n_iterations, termination_criterion=termination_criterion)
            self.W[D, j] = u[D]
            self.H[N, j] = (sigma * v[N]).ravel()
            twoD_mask = [[True if (D[i] and N[j]) else False for j in range(len(N))] for i in range(len(D))]
            twoD_mask = np.asarray(twoD_mask)
            X_copy[twoD_mask] = 0
            if np.linalg.norm(X_copy) == 0:  # if matrix becomes completely zero (its Frobenious norm is zero)
                index_truncate_W_and_H = j
                break
        if index_truncate_W_and_H is not None:
            if index_truncate_W_and_H != 0:  # index_truncate_W_and_H == 0 happens if matrix to be decomposed has one row or column, or is totally almost rank 1
                self.W = self.W[:, :index_truncate_W_and_H]
                self.H = self.H[:, :index_truncate_W_and_H]
            else:
                self.W = self.W[:, 0].reshape((-1,1))
                self.H = self.H[:, 0].reshape((-1,1))
        return self.W, self.H, index_truncate_W_and_H + 1

    def modified_R1_decompose(self, X):
        # return cluster of samples found by set N
        self.all_rows_in_submatrix = True
        self.gamma_bar = 10
        n_iterations = 10 ** 3
        termination_criterion = 1
        X_copy = np.copy(X)
        clusters_of_N = []
        k = -1
        while np.linalg.norm(X_copy) != 0:
            k = k + 1
            print('...Frobenius norm of X: ', np.linalg.norm(X_copy))
            D, N, u, v, sigma = self.extract_approximately_rankOne_submatrix(X=X_copy, n_iterations=n_iterations, termination_criterion=termination_criterion)
            samples_selected = [j for j in range(len(N)) if (N[j] is True)]
            clusters_of_N.append(samples_selected)
            twoD_mask = [[True if (D[i] and N[j]) else False for j in range(len(N))] for i in range(len(D))]
            twoD_mask = np.asarray(twoD_mask)
            X_copy[twoD_mask] = 0
            if np.linalg.norm(X_copy) == 0:  # if matrix becomes completely zero (its Frobenious norm is zero)
                break
        return clusters_of_N, k

    def extract_approximately_rankOne_submatrix(self, X, n_iterations=10 ** 8, termination_criterion=1):
        norms_of_columns = [np.linalg.norm(X[:, j]) for j in range(X.shape[1])]
        j_0 = np.argmax(np.asarray(norms_of_columns))
        D = [True for i in range(X.shape[0])]
        N = [j==j_0 for j in range(X.shape[1])]
        sigma = np.linalg.norm(X[:, j_0])
        u = X[:, j_0] * (1 / sigma)
        v = np.zeros((X.shape[1], 1))
        for counter in range(n_iterations):
            u_previous_iteration = u
            v_previous_iteration = v
            sigma_previous_iteration = sigma
            D_previous_iteration = D
            N_previous_iteration = N
            v_bar = (X[D, :].T).dot(u[D])
            N = [True if ((self.gamma_bar * (v_bar[j] ** 2)) - np.linalg.norm(X[D, j]) ** 2 > 0) else False for j in range(X.shape[1])]
            # it helped me for masking: https://stackoverflow.com/questions/42488400/python-nested-if-statements-with-list-comprehension
            v[N, :] = (v_bar[N] * (1/np.linalg.norm(v_bar[N]))).reshape((-1, 1))
            u_bar = (X[:, N].dot(v[N])).ravel()
            if self.all_rows_in_submatrix is False:
                D = [True if ((self.gamma_bar * (u_bar[i] ** 2)) - np.linalg.norm(X[i, N]) ** 2 > 0) else False for i in range(X.shape[0])]
            sigma = np.linalg.norm(u[D])
            u[D] = u_bar[D] * (1 / sigma)
            number_of_changes_in_D = sum(np.asarray(D) ^ np.asarray(D_previous_iteration))
            number_of_changes_in_N = sum(np.asarray(N) ^ np.asarray(N_previous_iteration))
            print('errors:')
            print(number_of_changes_in_D, number_of_changes_in_N, np.linalg.norm(sigma - sigma_previous_iteration),
                  np.linalg.norm(u - u_previous_iteration), np.linalg.norm(v - v_previous_iteration))
            if termination_criterion == 1:
                if number_of_changes_in_D == 0 and number_of_changes_in_N == 0 and np.linalg.norm(sigma - sigma_previous_iteration) < 1e-1:
                    break
            elif termination_criterion == 2:
                if number_of_changes_in_D == 0 and number_of_changes_in_N == 0:
                    break
            # if np.linalg.norm(u - u_previous_iteration) < 1e-4 and \
            #         np.linalg.norm(v - v_previous_iteration) < 1e-4:
            #     break
            # if np.linalg.norm(u - u_previous_iteration) < 1e-4 and \
            #         np.linalg.norm(v - v_previous_iteration) < 1e-4 and \
            #         np.linalg.norm(sigma - sigma_previous_iteration) < 1e-4:
            #     break
            # print(counter, np.linalg.norm(u - u_previous_iteration), np.linalg.norm(v - v_previous_iteration), np.linalg.norm(sigma - sigma_previous_iteration))
        return D, N, u, v, sigma

