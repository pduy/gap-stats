"""
File: gap_stats.py
Author: Duy Pham
Email: pduydl@gmail.com
Github: https://github.com/pduy
Description: This is a small code base for generating the gap statistics of a
general clustering algorithm which requires the number of clusters (k) as input.
These statistics will then be used to analyze the best number of clusters for a
particular dataset using the provided algorithm.

There are 2 examples given: KMeans and Gaussian mixture model.
"""


import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from collections import namedtuple


GapStat = namedtuple('GapStat', 'gap error')


def compute_log_wk(data, clustering_func):
    """
    The Wk statistic introduced in the paper.
    (https://statweb.stanford.edu/~gwalther/gap)

    Wk = pooled within-cluster sum of squares around the cluster means.
    Args:
        data (ndarray): data to do clustering
        clustering_func (python function): np.ndarray -> list of cluster indices


    Returns:
        float: log of Wk statistic
    """
    clusters = clustering_func(data)

    distances = []
    for cluster in np.unique(clusters):
        data_of_cluster = data[clusters == cluster]
        cluster_mean = data_of_cluster.mean(axis=0)
        total_distance_to_centroid = np.sum(
            np.linalg.norm(data_of_cluster - cluster_mean, axis=1))

        distances.append(total_distance_to_centroid)

    return np.log(sum(distances))


def generate_null_reference(data):
    """
    Generate a reference dataset of the same shape as data, where each column
    has a uniformly generated points between the lower and upper bounds of the
    real column.
    """
    # Bounding box: (min, max) of each column
    col_mins = np.min(data, axis=0)
    col_maxes = np.max(data, axis=0)

    return np.random.uniform(col_mins, col_maxes, data.shape)


def compute_reference_wk(data, n_iters, clustering_func):
    """
    Uniformly generate reference datasets in the bounding box and compute the
    mean wk statistic for those reference datasets.

    Args:
        data (np.ndarray): input data
        n_iters (int): Number of reference datasets to be generated
        clustering_func (function): np.ndarray -> list of cluster indices


    Returns:
        (float, float): mean and standard error of reference wk statistics
    """
    ref_wks = [
        compute_log_wk(generate_null_reference(data), clustering_func)
        for _ in range(n_iters)
    ]

    return np.mean(ref_wks), np.sqrt(1 + 1.0 / n_iters) * np.std(ref_wks)


def get_gap(data, n_iters, clustering_func):
    """
    Compute (1) gap statistic of 1 particular clustering paradigm given by the
    clustering_func

    Args:
        data (np.ndarray): input data.
        n_iters (int): number of null references to be generated.
        clustering_func (python function): np.ndarray -> list of cluster indices

    Returns:
        GapStat: gap statistic of the data under the clustering algorithm.
    """
    data_inertia = compute_log_wk(data, clustering_func)
    ref_inertia, error = compute_reference_wk(data, n_iters, clustering_func)

    return GapStat(gap=ref_inertia - data_inertia, error=error)


def get_gaps(data, max_k, cluster_algo, n_iters=10):
    """
    Compute the gap statistics for each value of k (number of clusters) from
    0 to max_k.

    Args:
        data (np.ndarray): input data
        max_k (int): maximum number of clusters
        cluster_algo (python function): an (possibly sklearn) clustering
            function which takes `n_clusters` as parameter and must provide the
            method fit_predict()
        n_iters (int): number of iterations (number of references to generate)

    Returns:
        list: gap statistics for each value of k
    """
    if max_k > len(data):
        max_k = len(data)

    return [
        get_gap(data, n_iters,
                cluster_algo(n_clusters=k).fit_predict)
        for k in range(1, max_k + 1)
    ]


def get_gaps_kmeans(data, max_k, n_iters=10):
    """Wrapper method, computing gap statistics for kmeans algorithm.
    Could also be an example of how to use the method `get_gaps()` for an 
    arbitrary clustering algorithm.

    Args:
        data (np.ndarray): input data
        max_k (int): maximum number of clusters
        n_iters (int): number of iterations (number of references to generate)

    Returns:
        list: gap statistics for each value of k
    """
    return get_gaps(
        data, max_k=max_k, cluster_algo=KMeans, n_iters=n_iters)


def get_gaps_gmm(data, max_k, n_iters=10):
    """Wrapper method, computing gap statistics for Gaussian mixture model.
    Could also be an example of how to use the method `get_gaps()` for an 
    arbitrary clustering algorithm.

    Args:
        data (np.ndarray): input data
        max_k (int): maximum number of clusters
        n_iters (int): number of iterations (number of references to generate)

    Returns:
        list: gap statistics for each value of k
    """

    # Need to wrap by lambda because GaussianMixture requires n_components
    # instead of n_clusters.
    return get_gaps(
        data,
        max_k=max_k,
        cluster_algo=lambda n_clusters: GaussianMixture(n_clusters),
        n_iters=n_iters)


def get_best_k(gaps, minimal_k=True):
    """Find the best number of k based on the gap statistics. The logic is based
    on the paper (https://statweb.stanford.edu/~gwalther/gap).

    Args:
        gaps (list): list of GapStat
        minimal_k (bool): If True, find the minimum number of k satisfying the
            condition gap[k] >= gap[k - 1] - error[k - 1]. If False, find the
            number of k which maximizes the gap statistic.

            True is prefer and is the default as we want to avoid overfitting.

    Returns:
        int: best number of clusters
    """
    if minimal_k:
        best_k = 1
        for k in range(1, len(gaps)):
            if gaps[k - 1].gap >= gaps[k].gap - gaps[k].error:
                best_k = k
                break
    else:
        best_k = np.argmax([g.gap for g in gaps]) + 1

    return best_k
