import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
from lightonml import OPU
from lightonml.internal.simulated_device import SimulatedOpuDevice
from pycle_gpu.cl_algo.simplified_algo import SimplifiedHierarchicalGmm
from sgd_comp_learning import sampling_frequencies, compute_sketch

from pycle.compressive_learning.torch.CLHS_dGMM import CLHS_dGMM
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
from pycle.utils.projectors import ProjectorNoProjection, ProjectorExactUnit2Norm, ProjectorClip, ProjectorLessUnit2Norm
import os

import numpy as np
import torch
from torch.nn import functional as f
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# from src.pycle_gpu.sketching.frequency_matrix import DenseFrequencyMatrix
# from src.pycle_gpu.cl_algo.optimization import ProjectorClip, ProjectorLessUnit2Norm




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
    # (theta, weights) = ckm_solver.current_sol
    centroids, sigma = theta[..., :dim], theta[..., -dim:]

    plt.figure(figsize=(5, 5))
    plt.title("Compressively learned centroids")
    plt.scatter(X[:, 0], X[:, 1], s=1, alpha=0.15)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=1000 * weights)
    plt.legend(["Data", "Centroids"])
    plt.show()

    from pycle.utils.metrics import SSE

    print("SSE: {}".format(SSE(X, centroids)))

