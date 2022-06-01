import numpy as np
import torch

from pycle.sketching import MatrixFeatureMap, drawFrequencies, computeSketch, pick_sketch_by_indice_in_aggregated_sketch


def test_pick_sketch_by_indice_in_aggregated_sketch():
    d = 2
    m = 5
    n = 1000
    n_sig = 10
    X = torch.from_numpy(np.random.randn(n, d))

    arr_sigma = np.logspace(-3, 0, n_sig)
    arr_sigma_inv_sqrt, directions, R = drawFrequencies("FoldedGaussian", d, m, arr_sigma, keep_splitted=True)

    Phi = MatrixFeatureMap("complexexponential", (arr_sigma_inv_sqrt, directions, R))

    z = computeSketch(X, Phi, verbose=False)

    for i_sig, sig in enumerate(arr_sigma_inv_sqrt):
        Phi = MatrixFeatureMap("complexexponential", (sig, directions, R))
        z_atom = computeSketch(X, Phi, verbose=False)

        sub_z = pick_sketch_by_indice_in_aggregated_sketch(z.numpy(), i_sig, m)

        assert np.isclose(z_atom.numpy(), sub_z).all(), f"{sig}, {i_sig}: different sketches"
