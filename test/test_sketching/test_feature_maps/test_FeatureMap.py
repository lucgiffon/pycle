from pycle.compressive_learning.torch.CLOMP_CKM import CLOMP_CKM
from pycle.sketching import MatrixFeatureMap
from pycle.sketching.frequency_sampling import drawFrequencies, multi_scale_frequency_sampling, \
    drawFrequencies_UniformRadius
import pytest
import numpy as np
import torch
import pycle
import matplotlib.pyplot as plt
from loguru import logger

from pycle.utils.datasets import generatedataset_GMM


@pytest.fixture
def dim():
    return 2

@pytest.fixture
def nb_clust():
    return 4

@pytest.fixture
def X(dim, nb_clust):
    np.random.seed(20)  # easy
    # np.random.seed(722233)
    nb_sample = 20000  # Number of samples we want to generate
    # We use the generatedataset_GMM method from pycle (we ask that the entries are <= 1, and imbalanced clusters)
    X = generatedataset_GMM(dim, nb_clust, nb_sample, normalize='l_inf-unit-ball', balanced=False)
    X = torch.from_numpy(X).double()
    return X




@pytest.fixture
def bounds(dim):
    # Bounds on the dataset, necessary for compressive k-means
    bounds = torch.tensor([-np.ones(dim), np.ones(dim)])  # We assumed the data is normalized between -1 and 1
    return bounds




def test_overproduce_and_choose(dim, nb_clust, bounds, X):
    Phi_emp = overproduce_and_choose(dim, nb_clust, X)
    z = pycle.sketching.computeSketch(X, Phi_emp)

    dct_adam = {
        "maxiter_inner_optimizations": 1000,
        "tol_inner_optimizations": 1e-9,
        "lr_inner_optimizations": 0.01,
        "beta_1": 0.9,
        "beta_2": 0.99,
        "opt_method_step_1": "adam",
        "opt_method_step_34": "nnls",
        "opt_method_step_5": "adam",
    }

    ckm_solver = CLOMP_CKM(phi=Phi_emp, nb_mixtures=nb_clust, bounds=bounds, sketch=z, show_curves=False, dct_opt_method=dct_adam)

    # Launch the CLOMP optimization procedure
    ckm_solver.fit_once()

    # Get the solution
    (theta, weights) = ckm_solver.current_sol
    centroids, sigma = theta[..., :dim], theta[..., -dim:]

    plt.figure(figsize=(5, 5))
    plt.title("Compressively learned centroids")
    plt.scatter(X[:, 0], X[:, 1], s=1, alpha=0.15)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=1000 * weights)
    plt.legend(["Data", "Centroids"])
    plt.show()

    from pycle.utils.metrics import SSE

    logger.info("SSE: {}".format(SSE(X, centroids)))
