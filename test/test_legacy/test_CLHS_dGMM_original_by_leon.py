import pytest
import matplotlib.pyplot as plt
from pycle_gpu.cl_algo.simplified_algo import SimplifiedHierarchicalGmm
from sgd_comp_learning import sampling_frequencies, compute_sketch
from pycle.utils.datasets import generatedataset_GMM
import numpy as np
import torch


# cleaning might deserve removing
@pytest.mark.skip(reason="Not usefull anymore")
def test_fit_once():
    np.random.seed(20)  # easy
    # np.random.seed(722233)
    dim = 2  # Dimension
    nb_clust = 4  # Number of Gaussians
    nb_sample = 20000  # Number of samples we want to generate
    # We use the generatedataset_GMM method from pycle (we ask that the entries are <= 1, and imbalanced clusters)
    X = generatedataset_GMM(dim, nb_clust, nb_sample, normalize='l_inf-unit-ball', balanced=False)
    X = torch.from_numpy(X).double()


    # Bounds on the dataset, necessary for compressive k-means
    bounds = torch.tensor([-np.ones(dim), np.ones(dim)])  # We assumed the data is normalized between -1 and 1


    # Pick the dimension m: 5*K*d is usually (just) enough in clustering (here m = 50)
    # sketch_dim = nb_clust * dim
    sketch_dim = 10 * nb_clust * dim

    freq_mat, sigma2_bar = sampling_frequencies(X, sketch_dim, False)

    sketch = compute_sketch(X, freq_mat)

    random_idx = torch.randint(nb_sample, (1,)).to(torch.long)
    random_atom = X[random_idx]
    solver = SimplifiedHierarchicalGmm(freq_mat, nb_clust, sketch, sigma2_bar, 10,
                                       2, 0.01, 0, 0.9,
                                       0.98, 1, random_atom, 1e-10, 0)
    solver.fit_once(".")


    weights, theta, sigmas_mat = solver.get_gmm()
    # Get the solution
    # (theta, weights) = ckm_solver.current_solution
    centroids, sigma = theta[..., :dim], theta[..., -dim:]

    plt.figure(figsize=(5, 5))
    plt.title("Compressively learned centroids")
    plt.scatter(X[:, 0], X[:, 1], s=1, alpha=0.15)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=1000 * weights)
    plt.legend(["Data", "Centroids"])
    plt.show()

    from pycle.utils.metrics import SSE

    print("SSE: {}".format(SSE(X, centroids)))

