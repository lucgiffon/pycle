import pycle.sketching
from pycle.compressive_learning.CLOMP_CKM import CLOMP_CKM
import pytest
import numpy as np
import torch
from loguru import logger
from pycle.sketching import MatrixFeatureMap
from pycle.utils.datasets import generatedataset_GMM
from pycle.utils.metrics import SSE
from pycle.utils.projectors import ProjectorClip
from pycle.utils.vizualization import simple_plot_clustering


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
    bounds = torch.tensor(np.array([-np.ones(dim), np.ones(dim)]))  # We assumed the data is normalized between -1 and 1
    return bounds


@pytest.fixture
def Phi_emp(nb_clust, dim):
    # For this simple example, assume we have a priori a rough idea of the size of the clusters
    Sigma = 0.1 * np.eye(dim)
    # Pick the dimension m: 5*K*d is usually (just) enough in clustering (here m = 50)
    sketch_dim = 10 * nb_clust * dim

    # According to the Folded Gaussian rule, we want m frequencies in dimension d, parametrized by Sigma
    Omega = pycle.sketching.frequency_sampling.drawFrequencies("FoldedGaussian", dim, sketch_dim, Sigma, return_torch=True)

    # The feature map is a standard one, the complex exponential of projections on Omega^T
    Phi_emp = MatrixFeatureMap("complexexponential", Omega, device=torch.device("cpu"))
    return Phi_emp


def test_fit_once_projector_clip(X, dim, nb_clust, bounds, Phi_emp):

    # Phi_gmm = GMMFeatureMap("None", Omega)

    # And sketch X with Phi: we map a 20000x2 dataset -> a 50-dimensional complex vector
    z = pycle.sketching.computeSketch(X, Phi_emp)
    # Initialize the solver object

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

    ckm_solver = CLOMP_CKM(phi=Phi_emp, size_mixture_K=nb_clust, bounds=bounds, sketch_z=z,
                           store_objective_values=False, dct_optim_method_hyperparameters=dct_adam,
                           centroid_projector=ProjectorClip(torch.tensor(-1), torch.tensor(1)))

    # Launch the CLOMP optimization procedure
    ckm_solver.fit_once()

    # Get the solution
    (theta, weights) = ckm_solver.current_solution
    centroids, sigma = theta[..., :dim], theta[..., -dim:]
    simple_plot_clustering(X, centroids, weights)
    logger.info("SSE: {}".format(SSE(X, centroids)))
