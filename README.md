# Python implementation of gap statistics for clustering analysis

This is a small code base for generating the gap statistics of a
general clustering algorithm which requires the number of clusters (k) as input.
These statistics will then be used to analyze the best number of clusters for a
particular dataset using the provided algorithm.

There are 2 examples given: KMeans and Gaussian mixture model.

Any clustering algorithm could be supported. You need to implement a clustering object which takes 
`n_clusters` as a parameter and provides the general sklearn prediction method `fit_predict()` which
returns a list of cluster indices from a data matrix.
