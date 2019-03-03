# Python implementation of gap statistics for clustering analysis

This is a small library for generating the gap statistics of a general
clustering algorithm which requires the number of clusters (k) as input. These
statistics will then be used to analyze the best number of clusters for a
particular dataset using the provided algorithm.

It is a reproduction of the original paper: https://statweb.stanford.edu/~gwalther/gap, with a generalization so that it could work with an arbitrary clustering algorithm.

Gap statistics is a well-known technique for analyzing clustering quality. It compares the clustering results from the real data with the results from uniformly sampled datasets within the bounding box of the real data. The minimum number of clusters which maximizes the distance between the two is considered as the appropriate number of clusters in the real data.

There are 2 examples given: KMeans and Gaussian mixture model.

Any clustering algorithm could be supported. We need to implement a clustering
class/object which takes `n_clusters` as a parameter and implements the general sklearn
prediction method `fit_predict()` which returns a list of cluster indices from a
data matrix.

## Usage

### Using supported clustering algorithms

K-Means and Gaussian-Mixture model are supported.

Suppose we have a `data` matrix and would like to compute the gap statistics
using K-Means for every `k` up to 10, with 100 iterations for each k:

```
import gap_stats
from sklearn.cluster import KMeans

gaps = gap_stats.get_gaps_kmeans(data, max_k=10, n_iters=100)
```

Each "gap" is a namedtuple GapStat(gap, error), where the two statistics are defined in the paper.

And from the statistics (gaps and errors) we would like to find the best k:

```
best_k = gap_stats.get_best_k(gaps, minimal_k=True)
```

### Using an arbitrary clustering algorithm

If we don't want to use K-Means or GMM, we could still use our own favorite clustering algorithm by passing it to the `get_gaps()` function.

Here is an example of "AgglomerativeClustering"

```
from sklearn.cluster import AgglomerativeClustering

gaps = gap_stats.get_gaps(x, max_k=10, cluster_algo=AgglomerativeClustering,
                          n_iters=10)
```

## Examples

More examples with visualization is included in the `/examples/experiment_gap_stats.ipynb` notebook.
