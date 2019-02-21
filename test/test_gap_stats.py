import pytest
import gap_stats
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs


@pytest.fixture
def generate_data():
    def _gen_data(n_features, n_centers):
        x, y = make_blobs(
            100,
            n_features=n_features,
            centers=n_centers,
            center_box=[-100, 100],
            random_state=1024)
        return x, y

    return _gen_data


@pytest.fixture
def simple_data():
    return np.array([[1, 2, 3], [4, 5, 6]])


@pytest.fixture
def simple_data_with_clustering():
    def simple_clustering_func(data):
        return [0, 0, 1, 1]

    return (np.array([[1, 2], [3, 2], [10, 11], [11, 11]]),
            simple_clustering_func)


def test_compute_log_wk(simple_data_with_clustering):
    data, clustering_func = simple_data_with_clustering
    log_wk = gap_stats.compute_log_wk(data, clustering_func)
    assert log_wk == 1.0986122886681098


def test_generate_null_reference(simple_data):
    null_ref = gap_stats.generate_null_reference(simple_data)

    assert null_ref.shape == simple_data.shape
    assert all(np.min(null_ref, axis=0) >= np.min(simple_data, axis=0))
    assert all(np.max(null_ref, axis=0) <= np.max(simple_data, axis=0))


@pytest.mark.parametrize('n_features, n_centers', [
    (2, 4),
    (2, 6),
    (3, 4),
    (3, 6)
])
def test_get_best_k(generate_data, n_features, n_centers):
    x, y = generate_data(n_features, n_centers)
    gaps = gap_stats.get_gaps(x, max_k=10, clustering_algorithm=KMeans, n_iters=100)

    best_k = gap_stats.get_best_k(gaps, minimal_k=True)
    assert best_k == n_centers
