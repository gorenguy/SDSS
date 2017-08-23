import numpy as np
import math
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing as mp
from functools import partial


def create2dobs(N):

    """
    :param N: Number of observations
    :return: (N,2) Array of observations - 2 clusters with noise
    """

    n1 = int(math.floor(0.65 * N))
    n2 = int(math.ceil(0.3 * N))
    n3=int(math.floor(0.05 * N))

    mean = [50, 60]
    cov = [[10, 40], [40, 200]]
    x1,y1 = np.random.multivariate_normal(mean,cov,n1).T

    mean = [65, 70]
    cov = [[20, 7], [7, 10]]
    x2,y2 = np.random.multivariate_normal(mean,cov,n2).T

    # and additional noises
    mean = [60, 60]
    cov = [[50, 0], [0, 50]]
    x3,y3 = np.random.multivariate_normal(mean,cov,n3).T

    # concatenate it all to a single vector
    x_total = np.concatenate((x1, x2, x3))
    y_total = np.concatenate((y1, y2, y3))

    X = np.array([x_total, y_total]).T

    return X,x_total,y_total


def plotSNE(sne):

    """
    :param sne: t-SNE object from sklearn.manifold.TSNE()
    :return: None. Function plots the t-SNE in 2d according to create2dobs() observations
    """

    N=len(sne)
    n1 = int(math.floor(0.65 * N))
    n2 = int(math.ceil(0.3 * N))
    n3 = int(math.floor(0.05 * N))

    plt.rcParams['figure.figsize'] = 5, 5
    plt.title("Data and outliers")
    plt.plot(sne[:n1 - 1, 0], sne[:n1 - 1, 1], "ok", label="group 1", markersize=4)
    plt.plot(sne[n1:n1 + n2 - 1, 0], sne[n1:n1 + n2 - 1:, 1], "oy", label="group 2", markersize=4)
    plt.plot(sne[n1 + n2:n1 + n2 + n3 - 1, 0], sne[n1 + n2:n1 + n2 + n3 - 1:, 1], "og", label="noise", markersize=4)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(loc="best")


def return_synthetic_data(X):
    """
    The function returns a matrix with the same dimensions as X but with synthetic data
    based on the marginal distributions of its featues
    """
    features = len(X[0])
    X_syn = np.zeros(X.shape)

    for i in xrange(features):
        obs_vec = X[:,i]
        syn_vec = np.random.choice(obs_vec, len(obs_vec)) # here we chose the synthetic data to match the marginal distribution of the real data
        X_syn[:,i] += syn_vec

    return X_syn


def merge_work_and_synthetic_samples(X, X_syn):

    """
    The function merges the data into one sample, giving the label "1" to the real data and label "2" to the synthetic data
    """

    # build the labels vector
    Y = np.ones(len(X))
    Y_syn = np.ones(len(X_syn)) * 2

    Y_total = np.concatenate((Y, Y_syn)) # Classes vector
    X_total = np.concatenate((X, X_syn)) # Merged array
    return X_total, Y_total


def calculateWnum(dis_mat, obj_ids, num):

    """
    :param dis_mat: Distance matrix
    :param obj_ids: A list containing the distance matrix object id's (i.e. 1:N)
    :param num: Number of closest galaxies for weirdness calculation
    :return: Weirdness score vector (Wnum)
    """

    w = np.zeros(len(obj_ids))
    for i in obj_ids:
        w[i] = np.sum(np.sort(dis_mat[i, :])[1:num + 1]) / float(num)

    return w


def calculateWall(dis_mat):

    """
    :param dis_mat: Distance matrix
    :return: Weirdness score vector (Wall)
    """

    wall = np.sum(dis_mat, axis=1)
    wall /= float(len(wall))

    return wall


def calcDisMat(rand_f, X):

    """
    :param rand_f: Random forest object - output of  sklearn.ensemble.RandomForestClassifier()
    :param X: (Number of obs. ,Number of feat.) Matrix
    :return: (Number of obs., Number of obs.) Distance matrix (1 - Similarity matrix)

    The function builds the similarity matrix based on the feature matrix X for the results Y
    based on the random forest we've trained
    the matrix is normalised so that the biggest similarity is 1 and the lowest is 0

    This function counts only leafs in which the object is classified as a "real" object.
    """

    n = len(X)

    # apply to get the leaf indices
    leafs = rand_f.apply(X)

    # find the predictions of the sample
    realLeafs = np.zeros(leafs.shape)
    for i, est in enumerate(rand_f.estimators_):
        realLeafs[:, i] = est.predict_proba(X)[:, 0] == True
    realLeafs = np.asarray(realLeafs, dtype=bool)

    # now calculate the similarity matrix
    sim_mat = np.zeros((n,n))
    for i in range(n):
        # Similarity = Number of Common Real Leafs / Number of Trees Both are Real Leafs
        sim_mat[i:,i] = (((leafs[i:,:] == leafs[i]) & realLeafs[i:,:]).sum(1)) / np.array(np.logical_and(realLeafs[i], realLeafs[i:,:]).sum(1), dtype=float)

    sim_mat += sim_mat.T - np.diag(np.diagonal(sim_mat))

    return 1 - sim_mat


def calcDisMat_par(rand_f, X):

    """
    :param rand_f: Random forest object - output of  sklearn.ensemble.RandomForestClassifier()
    :param X: (Number of obs. ,Number of feat.) Matrix
    :return: (Number of obs., Number of obs.) Distance matrix (1 - Similarity matrix)

    *** Parallel Version ***

    The function builds the similarity matrix based on the feature matrix X for the results Y
    based on the random forest we've trained
    the matrix is normalised so that the biggest similarity is 1 and the lowest is 0

    This function counts only leafs in which the object is classified as a "real" object.
    """
    n = len(X)

    # apply to get the leaf indices
    leafs = rand_f.apply(X)

    # find the predictions of the sample
    realLeafs = np.zeros(leafs.shape)
    for i, est in enumerate(rand_f.estimators_):
        realLeafs[:, i] = est.predict_proba(X)[:, 0] == True
    realLeafs = np.asarray(realLeafs, dtype=bool)

    # now calculate the similarity matrix
    sim_mat = np.zeros((n,n))

    pool = mp.Pool()
    f = partial(dist_i,leafs,realLeafs)
    res = pool.map(f, range(n))

    for tup in res:
        i = tup[1]
        sim_mat[i:, i] = tup[0]

    sim_mat += sim_mat.T - np.diag(np.diagonal(sim_mat))

    return 1 - sim_mat


def dist_i(leafs, realLeafs, i):
    # Similarity = Number of Common Real Leafs / Number of Trees Both are Real Leafs
    return((((leafs[i:, :] == leafs[i]) & realLeafs[i:, :]).sum(1)) / np.array(np.logical_and(realLeafs[i], realLeafs[i:, :]).sum(1), dtype=float), i)