import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
from pycle.legacy.CLHS_dGMM import CLHS_dGMM
from pycle.legacy.GMMFeatureMap import GMMFeatureMap
from pycle.sketching.feature_maps.MatrixFeatureMap import MatrixFeatureMap
from pycle.utils.datasets import generatedataset_GMM
from pycle.sketching.frequency_sampling import drawFrequencies

import pycle.sketching


@pytest.mark.skip(reason="Script not ready atm")
def test_fit_once():
    raise NotImplementedError("Test never worked.")
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

    # For this simple example, assume we have a priori a rough idea of the size of the clusters
    Sigma = 0.1 * np.eye(dim)

    # According to the Folded Gaussian rule, we want m frequencies in dimension d, parametrized by Sigma
    Omega = pycle.sketching.frequency_sampling.drawFrequencies("FoldedGaussian", dim, sketch_dim, Sigma, return_torch=True)

    # The feature map is a standard one, the complex exponential of projections on Omega^T
    Phi_emp = MatrixFeatureMap("ComplexExponential", Omega, use_torch=True, device=torch.device("cpu"))
    Phi_gmm = GMMFeatureMap("None", Omega, use_torch=True, device=torch.device("cpu"))

    # And sketch X with Phi: we map a 20000x2 dataset -> a 50-dimensional complex vector
    z = pycle.sketching.computeSketch(X, Phi_emp)
    # Initialize the solver object

    ckm_solver = CLHS_dGMM(phi=Phi_gmm, nb_mixtures=nb_clust, bounds=bounds, sketch=z, show_curves=True, sigma2_bar=0.1, random_atom=X[10], freq_batch_size=2,
                           maxiter_inner_optimizations=10)

    # Launch the CLOMP optimization procedure
    ckm_solver.fit_once()

    weights, theta, sigmas_mat = ckm_solver.get_gmm()
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

