import pytest
import torch
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from pycle.sketching.feature_maps.non_linearities import _universalQuantization, _universalQuantization_complex
from pycle.sketching.feature_maps.MatrixFeatureMap import MatrixFeatureMap
from pycle.utils.datasets import generatedataset_GMM
from pycle.compressive_learning.CLOMP_CKM import CLOMP_CKM
from pycle.utils.metrics import SSE

from pycle.sketching.frequency_sampling import drawFrequencies

import pycle.sketching
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
    X = generatedataset_GMM(dim, nb_clust, nb_sample, normalize='l_inf-unit-ball', imbalance=1/3)
    X = X.double()

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


@pytest.fixture
def Phi_emp_xi(nb_clust, dim):
    # For this simple example, assume we have a priori a rough idea of the size of the clusters
    Sigma = 0.1 * np.eye(dim)
    # Pick the dimension m: 5*K*d is usually (just) enough in clustering (here m = 50)
    sketch_dim = 100 * nb_clust * dim

    # According to the Folded Gaussian rule, we want m frequencies in dimension d, parametrized by Sigma
    Omega = pycle.sketching.frequency_sampling.drawFrequencies("FoldedGaussian", dim, sketch_dim, Sigma, return_torch=True)

    # The feature map is a standard one, the complex exponential of projections on Omega^T
    Phi_emp = MatrixFeatureMap("complexexponential", Omega, c_norm="unit", xi=torch.rand(sketch_dim) * np.pi * 2, device=torch.device("cpu"))
    return Phi_emp


def test_neg_floor_div_torch():
    neg_val = -4
    expected_result = neg_val // 3
    torch_result = torch.Tensor([neg_val]) // 3
    assert (expected_result == torch_result) == False


def test_fit_once_adam(X, dim, nb_clust, bounds, Phi_emp):

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

    ckm_solver = CLOMP_CKM(phi=Phi_emp, size_mixture_K=nb_clust, bounds=bounds, sketch_z=z, store_objective_values=False, dct_optim_method_hyperparameters=dct_adam)

    # Launch the CLOMP optimization procedure
    ckm_solver.fit_once()

    # Get the solution
    (theta, weights) = ckm_solver.current_solution
    centroids, sigma = theta[..., :dim], theta[..., -dim:]
    simple_plot_clustering(X, centroids, weights)
    logger.info("SSE: {}".format(SSE(X, centroids)))


def test_fit_once_bfgs(X, dim, nb_clust, bounds, Phi_emp):

    # Phi_gmm = GMMFeatureMap("None", Omega)

    # And sketch X with Phi: we map a 20000x2 dataset -> a 50-dimensional complex vector
    z = pycle.sketching.computeSketch(X, Phi_emp)
    # Initialize the solver object
    dct_bfgs = {
        "maxiter_inner_optimizations": 15000,
        "tol_inner_optimizations": 1e-9,
        "lr_inner_optimizations": 0.01,
        "opt_method_step_1": "lbfgs",
        "opt_method_step_34": "nnls",
        "opt_method_step_5": "lbfgs",
    }
    ckm_solver = CLOMP_CKM(phi=Phi_emp, size_mixture_K=nb_clust, bounds=bounds, sketch_z=z, store_objective_values=False, dct_optim_method_hyperparameters=dct_bfgs)

    # Launch the CLOMP optimization procedure
    ckm_solver.fit_once()

    # Get the solution
    (theta, weights) = ckm_solver.current_solution
    centroids, sigma = theta[..., :dim], theta[..., -dim:]

    simple_plot_clustering(X, centroids, weights)


    logger.info("SSE: {}".format(SSE(X, centroids)))


def test_fit_once_bfgs_xi(X, dim, nb_clust, bounds, Phi_emp_xi):

    # Phi_gmm = GMMFeatureMap("None", Omega)

    # And sketch X with Phi: we map a 20000x2 dataset -> a 50-dimensional complex vector
    z = pycle.sketching.computeSketch(X, Phi_emp_xi)
    # Initialize the solver object
    dct_bfgs = {
        "maxiter_inner_optimizations": 15000,
        "tol_inner_optimizations": 1e-9,
        "lr_inner_optimizations": 0.01,
        "opt_method_step_1": "lbfgs",
        "opt_method_step_34": "nnls",
        "opt_method_step_5": "lbfgs",
    }
    ckm_solver = CLOMP_CKM(phi=Phi_emp_xi, size_mixture_K=nb_clust, bounds=bounds, sketch_z=z, store_objective_values=False, dct_optim_method_hyperparameters=dct_bfgs)

    # Launch the CLOMP optimization procedure
    ckm_solver.fit_once()

    # Get the solution
    (theta, weights) = ckm_solver.current_solution
    centroids, sigma = theta[..., :dim], theta[..., -dim:]

    simple_plot_clustering(X, centroids, weights)
    logger.info("SSE: {}".format(SSE(X, centroids)))


def test_fit_once_pdfo(X, dim, nb_clust, bounds, Phi_emp):

    # Phi_gmm = GMMFeatureMap("None", Omega)

    # And sketch X with Phi: we map a 20000x2 dataset -> a 50-dimensional complex vector
    z = pycle.sketching.computeSketch(X, Phi_emp)
    # Initialize the solver object
    dct_pdfo = {
        "nb_iter_max_step_5": 500,
        "nb_iter_max_step_1": 500,
        "maxiter_inner_optimizations": 15000,
        "tol_inner_optimizations": 1e-9,
        "lr_inner_optimizations": 0.01,
        "opt_method_step_1": "pdfo",
        "opt_method_step_34": "nnls",
        "opt_method_step_5": "pdfo",
    }
    ckm_solver = CLOMP_CKM(phi=Phi_emp, size_mixture_K=nb_clust,
                           bounds=bounds, sketch_z=z, store_objective_values=False, dct_optim_method_hyperparameters=dct_pdfo)

    # Launch the CLOMP optimization procedure
    ckm_solver.fit_once()

    # Get the solution
    (theta, weights) = ckm_solver.current_solution

    simple_plot_clustering(X, theta, weights)

    logger.info("SSE: {}".format(SSE(X, theta)))


def test_plot_cosine_universal_q(dim, Phi_emp_xi):
    t = torch.tensor(np.linspace(-4 * np.pi, 4 * np.pi, 100))
    Phi_emp_xi.update_activation("cosine")
    f_t_cos = Phi_emp_xi.f(t)
    Phi_emp_xi.update_activation("universalquantization")
    f_t_uq = Phi_emp_xi.f(t)

    plt.figure(figsize=(5, 5))
    plt.plot(t.numpy(), f_t_cos.numpy())
    plt.plot(t.numpy(), f_t_uq.numpy())
    # plt.scatter(centroids[:, 0], centroids[:, 1], s=1000 * weights)
    plt.legend(["cos", "universal_q"])
    plt.show()


def test_universal_quantization():
    t = torch.tensor(np.linspace(-4 * np.pi, 4 * np.pi - 0.0000001, 10))

    f_t_uq_obtained = _universalQuantization(t)
    def _universal_quantization_truth(x): return torch.sign(torch.cos(x))
    f_t_uq_expected = _universal_quantization_truth(t)
    equality_real = f_t_uq_obtained == f_t_uq_expected
    assert equality_real.all()

    f_t_uq_obtained = _universalQuantization_complex(t)
    def _universal_quantization_truth(x): return torch.sign(torch.cos(x)) + 1.j * torch.sign(torch.sin(x))
    f_t_uq_expected = _universal_quantization_truth(t)
    equality_complex = f_t_uq_obtained == f_t_uq_expected
    assert equality_complex.all()


def test_asymetric(X, dim, nb_clust, bounds, Phi_emp_xi):
    # Phi_gmm = GMMFeatureMap("None", Omega)
    Phi_emp_xi.update_activation("universalquantization_complex")
    # And sketch X with Phi: we map adevice = {device} cpu 20000x2 dataset -> a 50-dimensional complex vector
    z = pycle.sketching.computeSketch(X, Phi_emp_xi)
    Phi_emp_xi.update_activation("ComplexExponential")

    # Initialize the solver object
    dct_bfgs = {
        "maxiter_inner_optimizations": 15000,
        "tol_inner_optimizations": 1e-9,
        "lr_inner_optimizations": 0.01,
        "opt_method_step_1": "lbfgs",
        "opt_method_step_34": "nnls",
        "opt_method_step_5": "lbfgs",
    }
    ckm_solver = CLOMP_CKM(phi=Phi_emp_xi, size_mixture_K=nb_clust, bounds=bounds, sketch_z=z, store_objective_values=False, dct_optim_method_hyperparameters=dct_bfgs)

    # Launch the CLOMP optimization procedure
    ckm_solver.fit_once()

    # Get the solution
    (theta, weights) = ckm_solver.current_solution
    centroids, sigma = theta[..., :dim], theta[..., -dim:]

    simple_plot_clustering(X, centroids, weights)

    logger.info("SSE: {}".format(SSE(X, centroids)))
