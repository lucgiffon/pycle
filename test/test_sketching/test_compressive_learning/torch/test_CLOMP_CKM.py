import pytest
import torch
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from lightonml import OPU
from lightonml.internal.simulated_device import SimulatedOpuDevice

from pycle.sketching.feature_maps.GMMFeatureMap import GMMFeatureMap
from pycle.sketching.feature_maps.MatrixFeatureMap import MatrixFeatureMap
from pycle.sketching.feature_maps.OPUFeatureMap import OPUFeatureMap
from pycle.utils.datasets import generatedataset_GMM
import pycle.sketching as sk
from pycle.compressive_learning.torch.CLOMP_CKM import CLOMP_CKM
from pycle.compressive_learning.numpy.CLOMP_CKM import CLOMP_CKM as CLOMP_CKM_NP

# from pycle.sketching.feature_maps.SimpleFeatureMap import SimpleFeatureMap

from pycle.sketching.frequency_sampling import drawFrequencies

import pycle.sketching
from pycle.utils.projectors import ProjectorNoProjection, ProjectorExactUnit2Norm


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

@pytest.fixture
def Phi_emp(nb_clust, dim):
    # For this simple example, assume we have a priori a rough idea of the size of the clusters
    Sigma = 0.1 * np.eye(dim)
    # Pick the dimension m: 5*K*d is usually (just) enough in clustering (here m = 50)
    sketch_dim = 10 * nb_clust * dim

    # According to the Folded Gaussian rule, we want m frequencies in dimension d, parametrized by Sigma
    Omega = pycle.sketching.frequency_sampling.drawFrequencies("FoldedGaussian", dim, sketch_dim, Sigma, use_torch=True)

    # The feature map is a standard one, the complex exponential of projections on Omega^T
    Phi_emp = MatrixFeatureMap("ComplexExponential", Omega, use_torch=True, device=torch.device("cpu"))
    return Phi_emp

def test_fit_once(X, dim, nb_clust, bounds, Phi_emp):

    # Phi_gmm = GMMFeatureMap("None", Omega)

    # And sketch X with Phi: we map a 20000x2 dataset -> a 50-dimensional complex vector
    z = pycle.sketching.computeSketch(X, Phi_emp)
    # Initialize the solver object

    ckm_solver = CLOMP_CKM(phi=Phi_emp, nb_mixtures=nb_clust, bounds=bounds, sketch=z, show_curves=False)

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


def test_fit_once_2(X, dim, nb_clust, bounds, Phi_emp):

    # Phi_gmm = GMMFeatureMap("None", Omega)

    # And sketch X with Phi: we map a 20000x2 dataset -> a 50-dimensional complex vector
    z = pycle.sketching.computeSketch(X, Phi_emp)
    # Initialize the solver object

    ckm_solver = CLOMP_CKM(phi=Phi_emp, nb_mixtures=nb_clust, bounds=bounds, sketch=z, show_curves=False, opt_method="lbfgs", maxiter_inner_optimizations=15000, tol_inner_optimizations=1e-9)

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