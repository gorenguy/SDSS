# Original
def build_similarity_matrix(rand_f, X):
    """
    The function builds the similarity matrix based on the feature matrix X for the results Y
    based on the random forest we've trained
    the matrix is normalised so that the biggest similarity is 1 and the lowest is 0

    This function counts only leafs in which the object is classified as a "real" object
    it is also implemented to optimize running time, asumming one has enough running memory
    """
    # apply to get the leaf indices
    apply_mat = rand_f.apply(X)
    # find the predictions of the sample
    is_good_matrix = np.zeros(apply_mat.shape)
    for i, est in enumerate(rand_f.estimators_):
        d = est.predict_proba(X)[:, 0] == 1
        is_good_matrix[:, i] = d
    # mark leafs that make the wrong prediction as -1, in order to remove them from the distance measurement
    apply_mat[is_good_matrix == False] = -1
    # now calculate the similarity matrix
    sim_mat = np.sum((apply_mat[:, None] == apply_mat[None, :]) & (apply_mat[:, None] != -1) & (apply_mat[None, :] != -1), axis=2) / np.asfarray(np.sum([apply_mat != -1], axis=2), dtype='float')

    return sim_mat

# Improved Performance
def build_similarity_matrix2(rand_f, X):
    """
    The function builds the similarity matrix based on the feature matrix X for the results Y
    based on the random forest we've trained
    the matrix is normalised so that the biggest similarity is 1 and the lowest is 0

    This function counts only leafs in which the object is classified as a "real" object
    it is also implemented to optimize running time, asumming one has enough running memory
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
    sim_mat = np.zeros((n, n))
    nRealLeafs = np.asfarray(np.sum(realLeafs, axis=1), dtype='float')
    for i in range(n):
        sim_mat[i] = ((leafs == leafs[i]) & realLeafs).sum(1) / nRealLeafs

    return sim_mat