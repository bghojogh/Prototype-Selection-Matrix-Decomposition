import pickle
import numpy as np
from sampleReduction_Decomposition import SR_MD
from sampleReduction_RecursiveMedian import SR_RecursiveMedian
from sampleReduction_edited_NN import SR_edited_NN
from sampleReduction_DROP import SR_Drop
from random import randint
import pandas as pd
import time
import os
from sklearn.model_selection import KFold
from sklearn.svm import SVC   # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
from sklearn.svm import LinearSVC   # http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA   # http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
from sklearn.ensemble import RandomForestClassifier as RF    # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA   # http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html
from sklearn.linear_model import LogisticRegression as LR   # http://nbviewer.jupyter.org/gist/justmarkham/6d5c061ca5aee67c4316471f8c2ae976   AND   http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.naive_bayes import GaussianNB   # http://scikit-learn.org/stable/modules/naive_bayes.html
from sklearn.neighbors import KNeighborsClassifier as KNN   # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn import preprocessing   # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
from IPython.display import display  # https://stackoverflow.com/questions/26873127/show-dataframe-as-table-in-ipython-notebook
from sklearn.linear_model import LinearRegression   # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
from sklearn.linear_model import Ridge   # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
from sklearn.linear_model import Lasso   # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
from sklearn.cluster import KMeans   # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
from sklearn.cluster import Birch   # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html#sklearn.cluster.Birch
from sklearn.cluster import SpectralClustering   # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html
from sklearn.cluster import DBSCAN    # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
from sklearn.metrics.cluster import adjusted_rand_score   # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
import random   # https://stackoverflow.com/questions/9755538/how-do-i-create-a-list-of-numbers-without-duplicates


def main():
    # ----- Settings:
    method = 'SVD_python'     # 'SVD_python', 'SVD_Jordan', 'SPCA', 'FLDA', 'R1D', 'modified_R1D',
                                # 'NMF_sklearn', 'Dictionary_learning', 'ICA', 'QR_decomposition', 'LU_decomposition',
                                # 'sorted_by_distance_from_mean', 'simple_random_sampling', 'stratified_sampling', 'recursive_median', 'ENN', 'DROP1'
    experiment = 'classification'   # 'unsupervised 2D demo', 'regression 2D demo', 'classification 2D demo', 'classification', 'regression', 'clustering'
    demo = False
    if demo is False:
        dataset = 'page_block'   # 'page_block', 'pima', 'spam', 'image_segmentation', 'yeast', 'wine', 'iris', 'fertility', 'balance', 'isolet', 'facebook', 'forest_fire', 'medical_cost'
        split_in_cross_validation_again = False
        shuffle_in_crossValidation = True
        rank_samples_again = False
        percentage_data = 20
        n_folds = 10
        n_repeats_inRandomSampling = 50
        # classifiers_for_experiments = ['NB', 'LDA', 'QDA', 'RF', 'LR', 'SVM']
        classifiers_for_experiments = ['KNN', 'NB', 'LDA', 'RF', 'LR', 'SVM']
        regressors_for_experiments = ['LinearRegression', 'Ridge', 'Lasso']
        clusteringMethods_for_experiments = ['KMeans', 'Birch', 'SpectralClustering', 'DBSCAN']
        base_path_to_save = './saved_files/' + experiment + '/' + dataset + '/' + method + '/percentage_remained=' + str(percentage_data) + '/'
    else:
        n_samples_demo = 1000
        n_samples_demo_classification = [100, 100, 100]
        n_samples_pick = None
        n_classes = 3
        base_path_to_save = './saved_files/' + experiment + '/' + method + '/'


    # ----- read input/synthetic data:
    if experiment == 'unsupervised 2D demo':
        # X = np.random.normal(size=[2, n_samples_demo])
        # X = np.random.normal(size=[2, n_samples_demo]) + 10 * np.ones((2, 1))
        mean = (10, 20)
        cov = [[10, 6], [5, 2]]
        X = np.random.multivariate_normal(mean, cov, size=n_samples_demo).T
        Y = None
    elif experiment == 'regression 2D demo':
        X = np.random.normal(size=[1, n_samples_demo])
        Y = np.random.normal(size=[1, n_samples_demo])
    elif experiment == 'classification 2D demo' or experiment == 'unsupervised 2D demo':
        # X = np.random.normal(size=[2, n_samples_demo])
        # Y = [randint(1, n_classes) for i in range(n_samples_demo)]
        # Y = np.asarray(Y).reshape((1,-1))
        n_dimensions = 2
        X = np.empty((n_dimensions, 0))
        Y = np.empty((1, 0))
        for class_index in range(n_classes):
            mean = np.zeros(n_dimensions)
            cov = np.eye(n_dimensions)
            if class_index == 0:
                mean = np.array([7, 5.5])
                cov = np.array([[5, 0], [0, 0.5]])
            elif class_index == 1:
                mean = np.array([4, 4])
                cov = np.array([[2, 0], [0, 2]])
            elif class_index == 2:
                mean = np.array([1.5, 2])
                cov = np.array([[0.7, 0], [0, 0.7]])
            X = np.hstack((X, create_Gaussian_samples(mean=mean, cov=cov, size=n_samples_demo_classification[class_index]).T))
            if experiment == 'classification 2D demo':
                Y = np.hstack((Y, np.ones((1, n_samples_demo_classification[class_index]))*class_index))
    elif experiment == 'classification' or experiment == 'regression' or experiment == 'clustering':
        if dataset == 'page_block':
            path_dataset = './input/page_block_dataset/page-blocks.data'
            X, Y = read_dataFormat_dataset(path_dataset, column_of_labels=-1)
            Y = Y.reshape((1, -1))
        elif dataset == 'pima':
            path_dataset = './input/pima_dataset/diabetes.csv'
            X, Y = read_csvFormat_dataset(path_dataset, column_of_labels=-1, sep=",", header='infer')
            Y = Y.reshape((1, -1))
        elif dataset == 'spam':
            path_dataset = './input/spam_dataset/spambase.txt'
            X, Y = read_csvFormat_dataset(path_dataset, column_of_labels=-1, sep=",", header=None)
            Y = Y.reshape((1, -1))
        elif dataset == 'image_segmentation':
            path_dataset = './input/image_segmentation_dataset/train_and_test_together.txt'
            X, Y = read_csvFormat_dataset(path_dataset, column_of_labels=0, sep=",", header=None)
            Y = Y.reshape((1, -1))
        elif dataset == 'yeast':
            path_dataset = './input/yeast_dataset/yeast.txt'
            X, Y = read_csvFormat_dataset(path_dataset, column_of_labels=-1, sep=" ", header=None)
            X = X.T; X = np.delete(X, 0, 1); X = X.T  # delete the first feature
            Y = Y.reshape((1, -1))
            X = np.array(X, dtype='float')
            Y = np.array(Y, dtype='float')  # otherwise, we get error
        elif dataset == 'wine':
            path_dataset = './input/wine_dataset/wine.txt'
            X, Y = read_csvFormat_dataset(path_dataset, column_of_labels=0, sep=",", header=None)
            Y = Y.reshape((1, -1))
        elif dataset == 'iris':
            path_dataset = './input/iris_dataset/iris.txt'
            X, Y = read_csvFormat_dataset(path_dataset, column_of_labels=-1, sep=",", header=None)
            Y = Y.reshape((1, -1))
        elif dataset == 'fertility':
            path_dataset = './input/fertility_dataset/fertility.txt'
            X, Y = read_csvFormat_dataset(path_dataset, column_of_labels=-1, sep=",", header=None)
            Y = Y.reshape((1, -1))
        elif dataset == 'balance':
            path_dataset = './input/balance_dataset/balance.txt'
            X, Y = read_csvFormat_dataset(path_dataset, column_of_labels=0, sep=",", header=None)
            Y = Y.reshape((1, -1))
        elif dataset == 'isolet':
            path_dataset = './input/isolet_dataset/isolet1+2+3+4.data'
            X1, Y1 = read_csvFormat_dataset(path_dataset, column_of_labels=-1, sep=",", header=None)
            Y1 = Y1.reshape((1, -1))
            path_dataset = './input/isolet_dataset/isolet5.data'
            X2, Y2 = read_csvFormat_dataset(path_dataset, column_of_labels=-1, sep=",", header=None)
            Y2 = Y2.reshape((1, -1))
            X = np.column_stack((X1, X2))
            Y = np.column_stack((Y1, Y2))
        elif dataset == 'facebook':
            path_dataset = './input/facebook_dataset/dataset_Facebook.csv'
            X, Y = read_facebook_dataset(path_dataset, sep=";", header='infer')
        elif dataset == 'forest_fire':
            path_dataset = './input/forest_fire_dataset/forestfires.csv'
            X, Y = read_regression_dataset_withOneOutput(path_dataset, column_of_labels=-1, sep=",", header='infer')
            Y = Y.reshape((1, -1))
        elif dataset == 'medical_cost':
            path_dataset = './input/medical_cost_dataset/insurance.csv'
            X, Y = read_regression_dataset_withOneOutput(path_dataset, column_of_labels=-1, sep=",", header='infer')
            Y = Y.reshape((1, -1))

    # ----- sample reduction:
    if experiment == 'unsupervised 2D demo' or experiment == 'regression 2D demo' or experiment == 'classification 2D demo':
        X_sorted, Y_sorted, scores_sorted = rank_samples(X=X, Y=Y, experiment=experiment, method=method)
        X_sorted, Y_sorted = reduce_samples(X_sorted=X_sorted, Y_sorted=Y_sorted, n_samples_pick=n_samples_pick)
    elif experiment == 'classification' or experiment == 'regression' or experiment == 'clustering':
        # ----- splitting data to train and test (cross validation):
        print('###### Splitting data for cross validation:')
        path_to_save = './saved_files/' + experiment + '/' + dataset + '/folds/'
        if split_in_cross_validation_again:
            train_indices_in_folds, test_indices_in_folds, X_train_in_folds, X_test_in_folds, y_train_in_folds, y_test_in_folds = cross_validation_kfold(X=X.T, y=Y.T, n_folds=n_folds, shuffle=shuffle_in_crossValidation)
            save_variable(train_indices_in_folds, 'train_indices_in_folds', path_to_save=path_to_save)
            save_variable(test_indices_in_folds, 'test_indices_in_folds', path_to_save=path_to_save)
            save_variable(X_train_in_folds, 'X_train_in_folds', path_to_save=path_to_save)
            save_variable(X_test_in_folds, 'X_test_in_folds', path_to_save=path_to_save)
            save_variable(y_train_in_folds, 'y_train_in_folds', path_to_save=path_to_save)
            save_variable(y_test_in_folds, 'y_test_in_folds', path_to_save=path_to_save)
        else:
            file = open(path_to_save+'train_indices_in_folds.pckl','rb')
            train_indices_in_folds = pickle.load(file); file.close()
            file = open(path_to_save+'test_indices_in_folds.pckl','rb')
            test_indices_in_folds = pickle.load(file); file.close()
            file = open(path_to_save+'X_train_in_folds.pckl','rb')
            X_train_in_folds = pickle.load(file); file.close()
            file = open(path_to_save+'X_test_in_folds.pckl','rb')
            X_test_in_folds = pickle.load(file); file.close()
            file = open(path_to_save+'y_train_in_folds.pckl','rb')
            y_train_in_folds = pickle.load(file); file.close()
            file = open(path_to_save+'y_test_in_folds.pckl','rb')
            y_test_in_folds = pickle.load(file); file.close()
        # ----- ranking samples:
        if method != 'simple_random_sampling' and method != 'stratified_sampling':
            print('###### Ranking data samples:')
            path_to_save = './saved_files/' + experiment + '/' + dataset + '/' + method + '/ranking/'
            if rank_samples_again:
                time_ranking = []
                X_sorted, Y_sorted, scores_sorted = [None] * n_folds, [None] * n_folds, [None] * n_folds
                for fold_index in range(n_folds):
                    print('############################## Cross validation: fold number ' + str(fold_index + 1) + ' out of ' + str(n_folds) + ' folds')
                    # --- taking the X and y of train and test sets for classification:
                    X_train = X_train_in_folds[fold_index]
                    X_test = X_test_in_folds[fold_index]
                    y_train = y_train_in_folds[fold_index]
                    y_test = y_test_in_folds[fold_index]
                    # --- rank samples:
                    start_timer = time.time()
                    if method != 'rank_other_methods':
                        X_sorted[fold_index], Y_sorted[fold_index], scores_sorted[fold_index] = rank_samples(X=X_train.T, Y=y_train.T, experiment=experiment, method=method)
                    end_timer = time.time()
                    time_ranking.extend([end_timer - start_timer])
                time_ranking_average = np.asarray(time_ranking).mean()
                time_ranking_std = np.asarray(time_ranking).std()
                save_variable(time_ranking, 'time_ranking', path_to_save=path_to_save)
                save_variable(time_ranking_average, 'time_ranking_average', path_to_save=path_to_save)
                save_variable(time_ranking_std, 'time_ranking_std', path_to_save=path_to_save)
                save_variable(X_sorted, 'X_sorted', path_to_save=path_to_save)
                save_variable(Y_sorted, 'Y_sorted', path_to_save=path_to_save)
                save_variable(scores_sorted, 'scores_sorted', path_to_save=path_to_save)
                save_np_array_to_txt(time_ranking, 'time_ranking', path_to_save=path_to_save)
                save_np_array_to_txt(time_ranking_average, 'time_ranking_average', path_to_save=path_to_save)
                save_np_array_to_txt(time_ranking_std, 'time_ranking_std', path_to_save=path_to_save)
            else:
                file = open(path_to_save + 'time_ranking_average.pckl', 'rb')
                time_ranking_average = pickle.load(file); file.close()
                file = open(path_to_save + 'time_ranking_std.pckl', 'rb')
                time_ranking_std = pickle.load(file); file.close()
                file = open(path_to_save + 'X_sorted.pckl', 'rb')
                X_sorted = pickle.load(file); file.close()
                file = open(path_to_save + 'Y_sorted.pckl', 'rb')
                Y_sorted = pickle.load(file); file.close()
                file = open(path_to_save + 'scores_sorted.pckl', 'rb')
                scores_sorted = pickle.load(file); file.close()
            # ----- reducing data:
            path_to_save = base_path_to_save + 'ranking/'
            if experiment == 'classification':
                print('###### Data reduction:')
                for fold_index in range(n_folds):
                    # --- taking the X and y of train and test sets for classification:
                    X_train = X_sorted[fold_index]
                    y_train = Y_sorted[fold_index]
                    # --- take a percentage of data:
                    X_separated_classes = SR_MD.separate_samples_of_classes(X=X_train.T, y=y_train.T)
                    n_samples_pick = []
                    for class_index in range(len(X_separated_classes)):
                        n_samples_of_class = X_separated_classes[class_index].shape[0]
                        n_samples_pick.extend([int(n_samples_of_class * percentage_data / 100)])
                    X_sorted[fold_index], Y_sorted[fold_index] = reduce_samples_classification(X_sorted=X_sorted[fold_index], Y_sorted=Y_sorted[fold_index], n_samples_pick=n_samples_pick)
                save_variable(n_samples_pick, 'n_samples_pick', path_to_save=path_to_save)
                save_np_array_to_txt(n_samples_pick, 'n_samples_pick', path_to_save=path_to_save)
                print('number of samples picked in each class: ', n_samples_pick)
                # ----- classification:
                print('###### Classifications:')
                recognition_rate_in_folds = do_classification(X_sorted=X_sorted, Y_sorted=Y_sorted, X_test_in_folds=X_test_in_folds, y_test_in_folds=y_test_in_folds, classifiers_for_experiments=classifiers_for_experiments, n_folds=n_folds)
            elif experiment == 'regression' or experiment == 'clustering':
                print('###### Data reduction:')
                for fold_index in range(n_folds):
                    n_samples_in_fold = X_sorted[fold_index].shape[1]
                    # --- take a percentage of data:
                    n_samples_pick = int(n_samples_in_fold * percentage_data / 100)
                    X_sorted[fold_index], Y_sorted[fold_index] = reduce_samples_regression_or_clustering(X_sorted=X_sorted[fold_index], Y_sorted=Y_sorted[fold_index], n_samples_pick=n_samples_pick)
                print('number of samples picked from dataset: ', n_samples_pick)
                # ----- regression/clustering:
                if experiment == 'regression':
                    print('###### Regressions:')
                    recognition_rate_in_folds = do_regression(X_sorted=X_sorted, Y_sorted=Y_sorted, X_test_in_folds=X_test_in_folds, y_test_in_folds=y_test_in_folds, regressors_for_experiments=regressors_for_experiments, n_folds=n_folds)
                elif experiment == 'clustering':
                    print('###### Clustering:')
                    recognition_rate_in_folds = do_clustering(X_sorted=X_sorted, Y_sorted=Y_sorted, X_test_in_folds=X_test_in_folds, y_test_in_folds=y_test_in_folds, clusteringMethods_for_experiments=clusteringMethods_for_experiments, n_folds=n_folds)
            # ----- saving results:
            path_to_save = base_path_to_save + '/accuracy/'
            save_variable(recognition_rate_in_folds, 'recognition_rate_in_folds', path_to_save=path_to_save)
            save_np_array_to_txt(recognition_rate_in_folds, 'recognition_rate_in_folds', path_to_save=path_to_save)
            # --- averaging the rates:
            recognition_rate_average = np.asarray(recognition_rate_in_folds).mean(axis=1)
            recognition_rate_std = np.asarray(recognition_rate_in_folds).std(axis=1)
            # --- saving results in folder:
            save_variable(recognition_rate_in_folds, 'recognition_rate_in_folds', path_to_save=path_to_save)
            save_variable(recognition_rate_average, 'recognition_rate_average', path_to_save=path_to_save)
            save_variable(recognition_rate_std, 'recognition_rate_std', path_to_save=path_to_save)
            save_np_array_to_txt(recognition_rate_in_folds, 'recognition_rate_in_folds', path_to_save=path_to_save)
            save_np_array_to_txt(recognition_rate_average, 'recognition_rate_average', path_to_save=path_to_save)
            save_np_array_to_txt(recognition_rate_std, 'recognition_rate_std', path_to_save=path_to_save)
            # --- report the results:
            print('The <average, std> of recognition rates (different classifiers/regressors/clustering methods) over folds:')
            print(recognition_rate_average, recognition_rate_std)
            print('The <average, std> of timing of ranking samples (in seconds):')
            print(time_ranking_average, time_ranking_std)
        else:    # if method == 'simple_random_sampling' or method == 'stratified_sampling'
            if experiment == 'classification':
                recognition_rate_in_folds_and_runs = np.zeros((len(classifiers_for_experiments), n_folds, n_repeats_inRandomSampling))
            elif experiment == 'regression':
                recognition_rate_in_folds_and_runs = np.zeros((len(regressors_for_experiments), n_folds, n_repeats_inRandomSampling))
            elif experiment == 'clustering':
                recognition_rate_in_folds_and_runs = np.zeros((len(clusteringMethods_for_experiments), n_folds, n_repeats_inRandomSampling))
            for repeat_index in range(n_repeats_inRandomSampling):
                print('### run # ', str(repeat_index), ':')
                if method == 'simple_random_sampling':
                    print('###### Data reduction:')
                    X_sorted, Y_sorted, scores_sorted = [None] * n_folds, [None] * n_folds, [None] * n_folds
                    for fold_index in range(n_folds):
                        # --- taking the X and y of train and test sets for classification:
                        X_train = X_train_in_folds[fold_index]
                        X_test = X_test_in_folds[fold_index]
                        y_train = y_train_in_folds[fold_index]
                        y_test = y_test_in_folds[fold_index]
                        # --- reduces samples:
                        n_training_samples = X_train.shape[0]
                        n_samples_pick = int(n_training_samples * percentage_data / 100)
                        selected_samples_indices = random.sample(range(n_training_samples), n_samples_pick)
                        X_sorted[fold_index] = X_train[selected_samples_indices, :].T
                        Y_sorted[fold_index] = y_train[selected_samples_indices, :].T
                        print('number of samples picked from dataset: ', n_samples_pick)
                    if experiment == 'classification':
                        # ----- classification:
                        print('###### Classifications:')
                        recognition_rate_in_folds = do_classification(X_sorted=X_sorted, Y_sorted=Y_sorted, X_test_in_folds=X_test_in_folds, y_test_in_folds=y_test_in_folds, classifiers_for_experiments=classifiers_for_experiments, n_folds=n_folds)
                    elif experiment == 'regression' or experiment == 'clustering':
                        # ----- regression/clustering:
                        if experiment == 'regression':
                            print('###### Regressions:')
                            recognition_rate_in_folds = do_regression(X_sorted=X_sorted, Y_sorted=Y_sorted, X_test_in_folds=X_test_in_folds, y_test_in_folds=y_test_in_folds, regressors_for_experiments=regressors_for_experiments, n_folds=n_folds)
                        elif experiment == 'clustering':
                            print('###### Clustering:')
                            recognition_rate_in_folds = do_clustering(X_sorted=X_sorted, Y_sorted=Y_sorted, X_test_in_folds=X_test_in_folds, y_test_in_folds=y_test_in_folds, clusteringMethods_for_experiments=clusteringMethods_for_experiments, n_folds=n_folds)
                elif method == 'stratified_sampling':
                    print('###### Random ranking data samples:')
                    X_sorted, Y_sorted, scores_sorted = [None] * n_folds, [None] * n_folds, [None] * n_folds
                    for fold_index in range(n_folds):
                        # --- taking the X and y of train and test sets for classification:
                        X_train = X_train_in_folds[fold_index]
                        X_test = X_test_in_folds[fold_index]
                        y_train = y_train_in_folds[fold_index]
                        y_test = y_test_in_folds[fold_index]
                        # --- rank samples:
                        X_sorted[fold_index], Y_sorted[fold_index], scores_sorted[fold_index] = rank_samples(
                            X=X_train.T, Y=y_train.T, experiment=experiment, method=method)
                    if experiment == 'classification':
                        # ----- reducing data:
                        print('###### Data reduction:')
                        for fold_index in range(n_folds):
                            # --- taking the X and y of train and test sets for classification:
                            X_train = X_sorted[fold_index]
                            y_train = Y_sorted[fold_index]
                            # --- take a percentage of data:
                            X_separated_classes = SR_MD.separate_samples_of_classes(X=X_train.T, y=y_train.T)
                            n_samples_pick = []
                            for class_index in range(len(X_separated_classes)):
                                n_samples_of_class = X_separated_classes[class_index].shape[0]
                                n_samples_pick.extend([int(n_samples_of_class * percentage_data / 100)])
                            X_sorted[fold_index], Y_sorted[fold_index] = reduce_samples_classification(X_sorted=X_sorted[fold_index], Y_sorted=Y_sorted[fold_index], n_samples_pick=n_samples_pick)
                        print('number of samples picked in each class: ', n_samples_pick)
                        # ----- classification:
                        print('###### Classifications:')
                        recognition_rate_in_folds = do_classification(X_sorted=X_sorted, Y_sorted=Y_sorted, X_test_in_folds=X_test_in_folds, y_test_in_folds=y_test_in_folds, classifiers_for_experiments=classifiers_for_experiments, n_folds=n_folds)
                    elif experiment == 'regression' or experiment == 'clustering':
                        print('###### Data reduction:')
                        for fold_index in range(n_folds):
                            n_samples_in_fold = X_sorted[fold_index].shape[1]
                            # --- take a percentage of data:
                            n_samples_pick = int(n_samples_in_fold * percentage_data / 100)
                            X_sorted[fold_index], Y_sorted[fold_index] = reduce_samples_regression_or_clustering(X_sorted=X_sorted[fold_index], Y_sorted=Y_sorted[fold_index], n_samples_pick=n_samples_pick)
                        print('number of samples picked from dataset: ', n_samples_pick)
                        # ----- regression/clustering:
                        if experiment == 'regression':
                            print('###### Regressions:')
                            recognition_rate_in_folds = do_regression(X_sorted=X_sorted, Y_sorted=Y_sorted, X_test_in_folds=X_test_in_folds, y_test_in_folds=y_test_in_folds, regressors_for_experiments=regressors_for_experiments, n_folds=n_folds)
                        elif experiment == 'clustering':
                            print('###### Clustering:')
                            recognition_rate_in_folds = do_clustering(X_sorted=X_sorted, Y_sorted=Y_sorted, X_test_in_folds=X_test_in_folds, y_test_in_folds=y_test_in_folds, clusteringMethods_for_experiments=clusteringMethods_for_experiments, n_folds=n_folds)
                recognition_rate_in_folds_and_runs[:, :, repeat_index] = np.asarray(recognition_rate_in_folds)
                # --- save results:
                path_to_save = base_path_to_save + '/accuracy/run_' + str(repeat_index) + '/'
                save_variable(recognition_rate_in_folds, 'recognition_rate_in_folds', path_to_save=path_to_save)
            # --- averaging the rates:
            recognition_rate_averageOverRuns = recognition_rate_in_folds_and_runs.mean(axis=2)
            recognition_rate_stdOverRuns = recognition_rate_in_folds_and_runs.std(axis=2)
            recognition_rate_averageOverFolds = recognition_rate_in_folds_and_runs.mean(axis=1)
            recognition_rate_stdOverFolds = recognition_rate_in_folds_and_runs.std(axis=1)
            recognition_rate_averageOverRunsAndFolds = recognition_rate_in_folds_and_runs.mean(axis=(1,2))  # https://stackoverflow.com/questions/18357065/get-mean-of-2d-slice-of-a-3d-array-in-numpy
            recognition_rate_stdOverRunsAndFolds = recognition_rate_in_folds_and_runs.std(axis=(1,2))
            # --- saving results in folder:
            path_to_save = base_path_to_save + '/accuracy/'
            save_variable(recognition_rate_in_folds_and_runs, 'recognition_rate_in_folds_and_runs', path_to_save=path_to_save)
            save_variable(recognition_rate_averageOverRuns, 'recognition_rate_averageOverRuns', path_to_save=path_to_save)
            save_variable(recognition_rate_stdOverRuns, 'recognition_rate_stdOverRuns', path_to_save=path_to_save)
            save_variable(recognition_rate_averageOverFolds, 'recognition_rate_averageOverFolds', path_to_save=path_to_save)
            save_variable(recognition_rate_stdOverFolds, 'recognition_rate_stdOverFolds', path_to_save=path_to_save)
            save_variable(recognition_rate_averageOverRunsAndFolds, 'recognition_rate_averageOverRunsAndFolds', path_to_save=path_to_save)
            save_variable(recognition_rate_stdOverRunsAndFolds, 'recognition_rate_stdOverRunsAndFolds', path_to_save=path_to_save)
            save_np_array_to_txt(recognition_rate_in_folds_and_runs, 'recognition_rate_in_folds_and_runs', path_to_save=path_to_save)
            save_np_array_to_txt(recognition_rate_averageOverRuns, 'recognition_rate_averageOverRuns', path_to_save=path_to_save)
            save_np_array_to_txt(recognition_rate_stdOverRuns, 'recognition_rate_stdOverRuns', path_to_save=path_to_save)
            save_np_array_to_txt(recognition_rate_averageOverFolds, 'recognition_rate_averageOverFolds', path_to_save=path_to_save)
            save_np_array_to_txt(recognition_rate_stdOverFolds, 'recognition_rate_stdOverFolds', path_to_save=path_to_save)
            save_np_array_to_txt(recognition_rate_averageOverRunsAndFolds, 'recognition_rate_averageOverRunsAndFolds', path_to_save=path_to_save)
            save_np_array_to_txt(recognition_rate_stdOverRunsAndFolds, 'recognition_rate_stdOverRunsAndFolds', path_to_save=path_to_save)
            # --- report the results:
            print('The <average, std> of recognition rates (different classifiers/regressors/clustering methods) over runs:')
            print(recognition_rate_averageOverRuns, recognition_rate_stdOverRuns)
            print('The <average, std> of recognition rates (different classifiers/regressors/clustering methods) over folds:')
            print(recognition_rate_averageOverFolds, recognition_rate_stdOverFolds)
            print('The <average, std> of recognition rates (different classifiers/regressors/clustering methods) over runs and folds:')
            print(recognition_rate_averageOverRunsAndFolds, recognition_rate_stdOverRunsAndFolds)

    # ----- plot results:
    path_to_save = base_path_to_save
    if experiment == 'unsupervised 2D demo':
        SR_MD.twoD_plot_samples_unsupervised(X=X, color_of_plot='b', axes_labels=['Dimension 1', 'Dimension 2'], marker_size=50, save_figures=True, path_save=path_to_save, name_save='samples', show_images=True)
        SR_MD.twoD_plot_sorted_samples_unsupervised(X_sorted=X_sorted, color_of_plot='b', axes_labels=['Dimension 1', 'Dimension 2'], biggest_marker_size=500, save_figures=True, path_save=path_to_save, name_save='sorted_samples', show_images=True)
    elif experiment == 'regression 2D demo':
        SR_MD.twoD_plot_samples_unsupervised(X=np.vstack((X,Y)), color_of_plot='b', axes_labels=['Observation', 'Label'], marker_size=50, save_figures=True, path_save=path_to_save, name_save='samples', show_images=True)
        SR_MD.twoD_plot_sorted_samples_unsupervised(X_sorted=np.vstack((X_sorted,Y_sorted)), color_of_plot='b', axes_labels=['Observation', 'Label'], biggest_marker_size=500, save_figures=True, path_save=path_to_save, name_save='sorted_samples', show_images=True)
    elif experiment == 'classification 2D demo':
        markers = ['o', 's', 'v']
        colors = ['b', 'r', 'g']
        SR_MD.twoD_plot_samples_classification(X_sorted=X_sorted, Y_sorted=Y_sorted, color_of_plot=colors, markers=markers, axes_labels=['Dimension 1', 'Dimension 2'], show_legends=True, marker_size=100, save_figures=True, path_save=path_to_save, name_save='samples', show_images=True)
        SR_MD.twoD_plot_sorted_samples_classification(X_sorted=X_sorted, Y_sorted=Y_sorted, color_of_plot=colors, markers=markers, axes_labels=['Dimension 1', 'Dimension 2'], show_legends=True, biggest_marker_size=500, save_figures=True, path_save=path_to_save, name_save='sorted_samples', show_images=True)


def rank_samples(X, Y, experiment, method):
    # input: X, Y --> rows: features, columns: samples
    # output: X_sorted, Y_sorted, scores_sorted --> rows: features, columns: samples
    if experiment == 'unsupervised 2D demo' or experiment == 'clustering':
        if method == 'SVD_python' or method == 'SVD_Jordan' or method == 'R1D' or method == 'modified_R1D' or method == 'NMF_sklearn' or method == 'Dictionary_learning' or method == 'ICA' or method == 'QR_decomposition' or method == 'LU_decomposition' or method == 'stratified_sampling':
            sr_md = SR_MD(X=X, Y=Y)
            X_sorted, Y_sorted, scores_sorted, scores = sr_md.find_scores_and_sort(method=method)
        elif method == 'sorted_by_distance_from_mean':
            sr_md_X = SR_MD(X=X, Y=None)
            X_sorted, _, scores_sorted, scores = sr_md_X.find_scores_and_sort(method=method)
            sr_md_Y = SR_MD(X=Y, Y=None)
            Y_sorted, _, _ = sr_md_Y._sort_samples(scores=scores)  # sort Y with the scores obtained from X
        elif method == 'recursive_median':
            sr_RecursiveMedian = SR_RecursiveMedian(X=X, Y=None, for_classification=False)
            X_sorted, _ = sr_RecursiveMedian.sort_samples_with_RecursiveMedian()
            Y_sorted = Y
            scores_sorted = None
    elif experiment == 'regression 2D demo' or experiment == 'regression':
        if method == 'SVD_python' or method == 'SVD_Jordan' or method == 'R1D' or method == 'modified_R1D' or method == 'NMF_sklearn' or method == 'Dictionary_learning' or method == 'ICA' or method == 'QR_decomposition' or method == 'LU_decomposition':
            sr_md_X = SR_MD(X=X)
            _, _, _, scores_X = sr_md_X.find_scores_and_sort(method=method)
            sr_md_Y = SR_MD(X=Y)
            _, _, _, scores_Y = sr_md_Y.find_scores_and_sort(method=method)
            scores = np.multiply(scores_X, scores_Y)
            X_sorted, _, scores_sorted = sr_md_X._sort_samples(scores=scores)
            Y_sorted, _, scores_sorted = sr_md_Y._sort_samples(scores=scores)
        elif method == 'SPCA':
            sr_md = SR_MD(X=X, Y=Y)
            X_sorted, Y_sorted, scores_sorted, scores = sr_md.find_scores_and_sort(method=method)
        elif method == 'sorted_by_distance_from_mean':
            sr_md_X = SR_MD(X=X, Y=None)
            X_sorted, _, scores_sorted, scores = sr_md_X.find_scores_and_sort(method=method)
            sr_md_Y = SR_MD(X=Y, Y=None)
            Y_sorted, _, _ = sr_md_Y._sort_samples(scores=scores)  # sort Y with the scores obtained from X
        elif method == 'stratified_sampling':
            sr_md = SR_MD(X=X, Y=Y) # we pass Y only to be sorted and we don't use it here
            X_sorted, Y_sorted, scores_sorted, scores = sr_md.find_scores_and_sort(method=method)
        elif method == 'recursive_median':
            sr_RecursiveMedian = SR_RecursiveMedian(X=X, Y=Y, for_classification=False) # we pass Y only to be sorted and we don't use it here
            X_sorted, Y_sorted = sr_RecursiveMedian.sort_samples_with_RecursiveMedian()
            scores_sorted = None
    elif experiment == 'classification 2D demo' or experiment == 'classification':
        if method == 'SPCA':
            sr_md = SR_MD(X=X, Y=Y)
            X_sorted, Y_sorted, scores_sorted, scores = sr_md.find_scores_and_sort(method=method)
        elif method == 'FLDA':
            sr_md = SR_MD(X=X, Y=Y)
            X_sorted, Y_sorted, scores_sorted, scores = sr_md.find_scores_and_sort(method=method)
        elif method == 'SVD_python' or method == 'SVD_Jordan' or method == 'R1D' or method == 'modified_R1D' or method == 'NMF_sklearn' or method == 'Dictionary_learning' or method == 'ICA' or method == 'QR_decomposition' or method == 'LU_decomposition':
            # --------> calculating scores considering within samples of classes:
            # put samples of each class next to each other:
            Y = np.asarray(Y)
            Y = Y.reshape((-1, 1))
            X = X.T
            yX = np.column_stack((Y, X))
            yX = yX[yX[:, 0].argsort()]  # sort array (asscending) with regards to nth column --> https://gist.github.com/stevenvo/e3dad127598842459b68
            Y = yX[:, 0].reshape((1, -1))  # ---> Y --> rows: features, columns: samples
            X = yX[:, 1:].T  # ---> X --> rows: features, columns: samples
            # separate samples of classes:
            X_separated_classes = SR_MD.separate_samples_of_classes(X=X.T, y=Y.T)
            # --------> calculating scores considering within classes:
            n_classes = len(X_separated_classes)
            scores = [None] * n_classes
            for class_index in range(n_classes):
                class_samples = X_separated_classes[class_index].T
                n_samples_of_class = class_samples.shape[1]
                class_labels = np.asarray([class_index for i in range(n_samples_of_class)]).reshape((1, -1))
                sr_md = SR_MD(X=class_samples, Y=class_labels)  # we pass class_labels only to be sorted and we don't use it here
                _, _, _, scores[class_index] = sr_md.find_scores_and_sort(method=method)
            scores_consideringWithinClasses = scores[0].ravel().tolist()
            for class_index in range(1, n_classes):
                scores_consideringWithinClasses.extend(scores[class_index].ravel().tolist())
            scores_consideringWithinClasses = np.asarray(scores_consideringWithinClasses).reshape((1, -1))
            # --------> calculating scores considering all samples with the labels:
            onehot_encoder = preprocessing.OneHotEncoder(sparse=False)  # https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
            Y_onehot_encoded = onehot_encoder.fit_transform(Y.T).T
            X_with_labels = np.vstack((X, Y_onehot_encoded))
            sr_md = SR_MD(X=X_with_labels, Y=Y)  # we pass Y only to be sorted and we don't use it here
            _, _, _, scores_consideringAllSamples = sr_md.find_scores_and_sort(method=method)
            # --------> fuse the two scores:
            scores_X = scores_consideringWithinClasses
            scores_D = scores_consideringAllSamples
            scores = np.multiply(scores_X, scores_D)
            X_with_labels_sorted, Y_sorted, scores_sorted = sr_md._sort_samples(scores=scores)
            X_sorted = X_with_labels_sorted[:X.shape[0], :]
        elif method == 'sorted_by_distance_from_mean':
            sr_md = SR_MD(X=X, Y=Y)
            X_sorted, Y_sorted, scores_sorted, scores = sr_md.find_scores_and_sort(method=method)
        elif method == 'stratified_sampling':
            sr_md = SR_MD(X=X, Y=Y) # we pass Y only to be sorted and we don't use it here
            X_sorted, Y_sorted, scores_sorted, scores = sr_md.find_scores_and_sort(method=method)
        elif method == 'recursive_median':
            sr_RecursiveMedian = SR_RecursiveMedian(X=X, Y=Y, for_classification=True)
            X_sorted, Y_sorted = sr_RecursiveMedian.sort_samples_with_RecursiveMedian()
            scores_sorted = None
        elif method == 'ENN':
            sr_edited_NN = SR_edited_NN(X=X, Y=Y, n_neighbors=5 + 1)  # note: in n_neighbors, the sample itself is counted
            X_sorted, Y_sorted, scores_sorted = sr_edited_NN.sort_samples_withENN()
        elif method == 'DROP1':
            sr_Drop = SR_Drop(X=X, Y=Y, n_neighbors=5 + 1)  # note: in n_neighbors, the sample itself is counted
            X_sorted, Y_sorted = sr_Drop.sort_samples_withDrop1_faster()
            # X_sorted, Y_sorted = sr_Drop.sort_samples_withDrop1()
            scores_sorted = None
    return X_sorted, Y_sorted, scores_sorted

def reduce_samples(X_sorted, Y_sorted, n_samples_pick):
    # input and output: X, Y --> rows: features, columns: samples
    n_samples = X_sorted.shape[1]
    if n_samples_pick is not None:
        n_samples_pick = min(n_samples_pick, n_samples)
        X_reduced = X_sorted[:, :n_samples_pick]
        if Y_sorted is not None:
            Y_reduced = Y_sorted[:, :n_samples_pick]
    else:
        X_reduced = X_sorted
        Y_reduced = Y_sorted
    return X_reduced, Y_reduced

def reduce_samples_classification(X_sorted, Y_sorted, n_samples_pick):
    # input and output: X, Y --> rows: features, columns: samples
    # n_samples_pick --> a list of integers
    n_samples = X_sorted.shape[1]
    n_classes = len(n_samples_pick)
    n_samples_pick = [(min(n_samples_pick[i], n_samples)) if (n_samples_pick[i] is not None) else n_samples for i in range(n_classes)]
    X_and_Y_sorted = np.vstack((X_sorted, Y_sorted))
    X_and_Y_sorted_separated_classes = separate_samples_of_classes_withoutChangingOrder(X=X_and_Y_sorted.T, y=Y_sorted.T)
    n_dimensions = X_sorted.shape[0]
    X_reduced = np.empty((n_dimensions, 0))
    Y_reduced = np.empty((Y_sorted.shape[0], 0))
    scores_sorted = np.empty((1, 0))
    for class_index in range(n_classes):
        X_and_Y_sorted_class = X_and_Y_sorted_separated_classes[class_index].T
        X_sorted_class = X_and_Y_sorted_class[:n_dimensions, :n_samples_pick[class_index]]
        Y_sorted_class = X_and_Y_sorted_class[n_dimensions:, :n_samples_pick[class_index]]
        X_reduced = np.column_stack((X_reduced, X_sorted_class))
        Y_reduced = np.column_stack((Y_reduced, Y_sorted_class))
    return X_reduced, Y_reduced

def reduce_samples_regression_or_clustering(X_sorted, Y_sorted, n_samples_pick):
    # input and output: X, Y --> rows: features, columns: samples
    # n_samples_pick --> a list of integers
    n_samples = X_sorted.shape[1]
    if n_samples_pick is not None:
        n_samples_pick = min(n_samples_pick, n_samples)
    else:
        n_samples_pick = n_samples
    X_reduced = X_sorted[:, :n_samples_pick]
    Y_reduced = Y_sorted[:, :n_samples_pick]
    return X_reduced, Y_reduced

def do_classification(X_sorted, Y_sorted, X_test_in_folds, y_test_in_folds, classifiers_for_experiments, n_folds):
    recognition_rate_in_folds = np.zeros((len(classifiers_for_experiments), n_folds))
    for classifier_index in range(len(classifiers_for_experiments)):
        classifier = classifiers_for_experiments[classifier_index]
        print('############# Classifier: ' + classifier)
        for fold_index in range(n_folds):
            print('############################## Cross validation: fold number ' + str(
                fold_index + 1) + ' out of ' + str(n_folds) + ' folds')
            X_train = X_sorted[fold_index].T
            X_test = X_test_in_folds[fold_index]
            y_train = (Y_sorted[fold_index].T).ravel()
            y_test = y_test_in_folds[fold_index].ravel()
            if classifier == 'SVM':
                # --------- train:
                # clf = SVC(kernel='linear')
                clf = LinearSVC(random_state=0)
                clf.fit(X=X_train, y=y_train)
            elif classifier == 'LDA':
                # --------- train:
                clf = LDA()
                clf.fit(X=X_train, y=y_train)
            elif classifier == 'QDA':
                # --------- train:
                clf = QDA()
                clf.fit(X=X_train, y=y_train)
            elif classifier == 'RF':
                # --------- train:
                clf = RF(max_depth=2, random_state=0)
                clf.fit(X=X_train, y=y_train)
            elif classifier == 'LR':
                # --------- train:
                clf = LR()
                clf.fit(X=X_train, y=y_train)
            elif classifier == 'NB':
                # --------- train:
                clf = GaussianNB()
                clf.fit(X=X_train, y=y_train)
            elif classifier == 'KNN':
                # --------- train:
                clf = KNN(n_neighbors=1)
                clf.fit(X=X_train, y=y_train)
            # --------- test:
            labels_predicted = clf.predict(X_test)
            recognition_rate = (sum(labels_predicted == y_test) / len(labels_predicted)) * 100
            print('The recognition rate using ' + classifier + ' in fold ' + str(fold_index + 1) + ': ' + str(
                recognition_rate))
            recognition_rate_in_folds[classifier_index, fold_index] = recognition_rate
        print('The average recognition rate using ' + classifier + ': ' + str(recognition_rate_in_folds[classifier_index, :].mean()) + ' +- ' + str(recognition_rate_in_folds[classifier_index, :].std()))
    return recognition_rate_in_folds

def do_regression(X_sorted, Y_sorted, X_test_in_folds, y_test_in_folds, regressors_for_experiments, n_folds):
    recognition_rate_in_folds = np.zeros((len(regressors_for_experiments), n_folds))
    for regressor_index in range(len(regressors_for_experiments)):
        regressor = regressors_for_experiments[regressor_index]
        print('############# Regressor: ' + regressor)
        for fold_index in range(n_folds):
            print('############################## Cross validation: fold number ' + str(fold_index + 1) + ' out of ' + str(n_folds) + ' folds')
            X_train = X_sorted[fold_index].T
            X_test = X_test_in_folds[fold_index]
            y_train = Y_sorted[fold_index].T
            y_test = y_test_in_folds[fold_index]
            if regressor == 'LinearRegression':
                # --------- train:
                clf = LinearRegression()
                clf.fit(X=X_train, y=y_train)
            elif regressor == 'Ridge':
                # --------- train:
                clf = Ridge(alpha=1.0)
                clf.fit(X=X_train, y=y_train)
            elif regressor == 'Lasso':
                # --------- train:
                clf = Lasso(alpha=1.0)
                clf.fit(X=X_train, y=y_train)
            # --------- test:
            labels_predicted = clf.predict(X=X_test)
            recognition_rate = clf.score(X=X_test, y=y_test)
            print('The recognition rate using ' + regressor + ' in fold ' + str(fold_index + 1) + ': ' + str(recognition_rate))
            recognition_rate_in_folds[regressor_index, fold_index] = recognition_rate
        print('The average recognition rate using ' + regressor + ': ' + str(recognition_rate_in_folds[regressor_index, :].mean()) + ' +- ' + str(recognition_rate_in_folds[regressor_index, :].std()))
    return recognition_rate_in_folds

def do_clustering(X_sorted, Y_sorted, X_test_in_folds, y_test_in_folds, clusteringMethods_for_experiments, n_folds):
    # clustering methods in python: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
    # clustering tutorial and evaluation metrics: http://scikit-learn.org/stable/modules/clustering.html
    recognition_rate_in_folds = np.zeros((len(clusteringMethods_for_experiments), n_folds))
    for clusteringMethod_index in range(len(clusteringMethods_for_experiments)):
        clusteringMethod = clusteringMethods_for_experiments[clusteringMethod_index]
        print('############# Clustering method: ' + clusteringMethod)
        for fold_index in range(n_folds):
            print('############################## Cross validation: fold number ' + str(fold_index + 1) + ' out of ' + str(n_folds) + ' folds')
            X_train = X_sorted[fold_index].T
            X_test = X_test_in_folds[fold_index]
            y_train = Y_sorted[fold_index].T
            y_test = y_test_in_folds[fold_index]
            y_trainAndTest = np.vstack((y_train, y_test))
            labels_of_classes = sorted(set(y_trainAndTest.ravel().tolist()))
            n_classes = len(labels_of_classes)
            if clusteringMethod == 'KMeans':
                # --------- train:
                clf = KMeans(n_clusters=n_classes)
                clf.fit(X=X_train)
                # --------- test:
                labels_predicted = clf.predict(X=X_test)
            elif clusteringMethod == 'Birch':
                # --------- train:
                clf = Birch(n_clusters=n_classes)
                clf.fit(X=X_train)
                # --------- test:
                labels_predicted = clf.predict(X=X_test)
            elif clusteringMethod == 'SpectralClustering':
                # --------- train/test:
                clf = SpectralClustering(n_clusters=n_classes)
                labels_predicted = clf.fit_predict(X=X_test)
            elif clusteringMethod == 'DBSCAN':
                # --------- train/test:
                clf = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
                labels_predicted = clf.fit_predict(X=X_test)
            recognition_rate = adjusted_rand_score(labels_true=y_test.ravel(), labels_pred=labels_predicted)
            print('The recognition rate using ' + clusteringMethod + ' in fold ' + str(fold_index + 1) + ': ' + str(recognition_rate))
            recognition_rate_in_folds[clusteringMethod_index, fold_index] = recognition_rate
        print('The average recognition rate using ' + clusteringMethod + ': ' + str(recognition_rate_in_folds[clusteringMethod_index, :].mean()) + ' +- ' + str(recognition_rate_in_folds[clusteringMethod_index, :].std()))
    return recognition_rate_in_folds

def create_Gaussian_samples(mean=np.zeros(2), cov=np.eye(2), size=5):
    # https://stackoverflow.com/questions/34932499/draw-multiple-samples-using-numpy-random-multivariate-normalmean-cov-size
    return np.random.multivariate_normal(mean=mean, cov=cov, size=size)

def read_dataFormat_dataset(path_dataset, column_of_labels):
    # return: X --> rows: features, columns = samples, Y --> vector of labels
    # https://stackoverflow.com/questions/30762762/convert-data-file-to-csv
    with open(path_dataset) as input_file:
        lines = input_file.readlines()
        newLines = []
        for line in lines:
            newLine = line.strip().split()
            newLines.append(newLine)
    newLines = np.asarray(newLines)
    newLines = newLines.astype(float)
    data = newLines
    try:
        y = data[:, column_of_labels].astype(int)
    except: # if is string
        y = data[:, column_of_labels]
        le = preprocessing.LabelEncoder()
        le.fit(y)
        y = le.transform(y)
    X = np.delete(data, column_of_labels, 1)  # delete the column of labels from X
    try:
        X = X.astype(np.float64)  # if we don't do that, we will have this error: https://www.reddit.com/r/learnpython/comments/7ivopz/numpy_getting_error_on_matrix_inverse/
    except:
        pass
    X = X.T
    return X, y

def read_csvFormat_dataset(path_dataset, column_of_labels, sep=",", header='infer'):
    # return: X --> rows: features, columns = samples, Y --> vector of labels
    if sep == ' ':
        data = pd.read_csv(path_dataset, delim_whitespace=True, header=header)  # https://stackoverflow.com/questions/19632075/how-to-read-file-with-space-separated-values
    else:
        data = pd.read_csv(path_dataset, sep=sep, header=header)  # read text file using pandas dataFrame: https://stackoverflow.com/questions/21546739/load-data-from-txt-with-pandas
    data = data.values  # converting pandas dataFrame to numpy array
    try:
        y = data[:, column_of_labels].astype(int)
    except: # if is string
        y = data[:, column_of_labels]
        le = preprocessing.LabelEncoder()
        le.fit(y)
        y = le.transform(y)
    X = np.delete(data, column_of_labels, 1)  # delete the column of labels from X
    try:
        X = X.astype(np.float64)  # if we don't do that, we will have this error: https://www.reddit.com/r/learnpython/comments/7ivopz/numpy_getting_error_on_matrix_inverse/
    except:
        pass
    X = X.T
    return X, y

def read_facebook_dataset(path_dataset, sep=";", header='infer'):
    # return: X, Y --> rows: features, columns = samples
    data = pd.read_csv(path_dataset, sep=sep, header=header)
    # print(list(data.columns.values))
    names_of_input_features = ["Category", "Page total likes", "Type", "Post Month", "Post Hour", "Post Weekday", "Paid"]
    names_of_output_features = ["Lifetime Post Total Reach", "Lifetime Post Total Impressions", "Lifetime Engaged Users", "Lifetime Post Consumers", "Lifetime Post Consumptions", "Lifetime Post Impressions by people who have liked your Page", "Lifetime Post reach by people who like your Page", "Lifetime People who have liked your Page and engaged with your post", "comment", "like", "share", "Total Interactions"]
    X = np.zeros((data.shape[0], len(names_of_input_features)))
    for feature_index, feature_name in enumerate(names_of_input_features):
        try:
            X[:, feature_index] = data.loc[:, feature_name].values
        except:  # feature if categorical
            feature_vector = data.loc[:, feature_name]
            le = preprocessing.LabelEncoder()
            le.fit(feature_vector)
            X[:, feature_index] = le.transform(feature_vector)
    X = X.T
    Y = np.zeros((data.shape[0], len(names_of_output_features)))
    for feature_index, feature_name in enumerate(names_of_output_features):
        Y[:, feature_index] = data.loc[:, feature_name].values
    Y = Y.T
    # Five samples have some nan values, such as: the "Paid" feature of last sample (X[-1,-1]) is missing and thus nan. We remove it:
    indices_of_samples_not_having_missing_values = np.logical_and(~np.isnan(X).any(axis=0), ~np.isnan(Y).any(axis=0))  # https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-91.php
    X = X[:, indices_of_samples_not_having_missing_values]
    Y = Y[:, indices_of_samples_not_having_missing_values]
    return X, Y

def read_regression_dataset_withOneOutput(path_dataset, column_of_labels, sep=";", header='infer'):
    # return: X, Y --> rows: features, columns = samples
    data = pd.read_csv(path_dataset, sep=sep, header=header)
    X = np.zeros(data.shape)
    for feature_index in range(data.shape[1]):
        try:
            X[:, feature_index] = data.iloc[:, feature_index].values
        except:  # feature if categorical
            feature_vector = data.iloc[:, feature_index]
            le = preprocessing.LabelEncoder()
            le.fit(feature_vector)
            X[:, feature_index] = le.transform(feature_vector)
    Y = X[:, column_of_labels]
    X = np.delete(X, column_of_labels, 1)  # delete the column of labels from X
    X = X.T
    Y = Y.T
    return X, Y

def save_variable(variable, name_of_variable, path_to_save='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.pckl'
    f = open(file_address, 'wb')
    pickle.dump(variable, f)
    f.close()

def save_np_array_to_txt(variable, name_of_variable, path_to_save='./'):
    if type(variable) is list:
        variable = np.asarray(variable)
    # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.txt'
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
    with open(file_address, 'w') as f:
        f.write(np.array2string(variable, separator=', '))

def separate_samples_of_classes_withoutChangingOrder(X, y):
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
    # for sample_index in range(n_samples):
    #     label_of_sample = y[sample_index]
    #     index_of_class_of_sample = [i for i, x in enumerate(labels_of_classes) if x == label_of_sample]  # https://stackoverflow.com/questions/364621/how-to-get-items-position-in-a-list
    #     index_of_class_of_sample = index_of_class_of_sample[0]
    #     X_separated_classes[index_of_class_of_sample] = np.vstack([X_separated_classes[index_of_class_of_sample], X[sample_index:, :]])
    #     print(sample_index, n_samples)
    return X_separated_classes

def cross_validation_kfold(X, y, n_folds=10, shuffle=False):
    # input: X, y --> rows are samples and columns are features
    # return: --> rows are samples and columns are features
    # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit
    kf = KFold(n_splits=n_folds, shuffle=shuffle)
    train_indices_in_folds = []; test_indices_in_folds = []
    X_train_in_folds = []; X_test_in_folds = []
    y_train_in_folds = []; y_test_in_folds = []
    for train_index, test_index in kf.split(X):
        train_indices_in_folds.append(train_index)
        test_indices_in_folds.append(test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np.asarray(y)[train_index], np.asarray(y)[test_index]
        X_train_in_folds.append(X_train)
        X_test_in_folds.append(X_test)
        y_train_in_folds.append(y_train)
        y_test_in_folds.append(y_test)
    return train_indices_in_folds, test_indices_in_folds, X_train_in_folds, X_test_in_folds, y_train_in_folds, y_test_in_folds

if __name__ == '__main__':
    main()