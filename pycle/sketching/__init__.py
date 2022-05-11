"""
Sketching functions
"""
import sys

import numpy as np
import torch

from loguru import logger

from pycle.sketching.feature_maps.FeatureMap import FeatureMap
from pycle.sketching.feature_maps.MatrixFeatureMap import MatrixFeatureMap
from pycle.utils import is_number


def computeSketch(dataset, featureMap, datasetWeights=None, batch_size=100, display=True):
    """
    Computes the sketch of a dataset given a generic feature map.

    More precisely, evaluates
        z = sum_{x_i in X} w_i * Phi(x_i)
    where X is the dataset, Phi is the sketch feature map, w_i are weights assigned to the samples (typically 1/n).

    Arguments:
        - dataset        : (n,d) numpy array, the dataset X: n examples in dimension d
        - featureMap     : the feature map Phi, given as one of the following:
            -- a function, z_x_i = featureMap(x_i), where x_i and z_x_i are (n,)- and (m,)-numpy arrays, respectively
            -- a FeatureMap instance (e.g., constructed as featureMap = SimpleFeatureMap("complexExponential",Omega) )
        - datasetWeights : (n,) numpy array, optional weigths w_i in the sketch (default: None, corresponds to w_i = 1/n)

    Returns:
        - sketch : (m,) numpy array, the sketch as defined above
    """
    # TODOs:
    # - add possibility to specify classes and return one sketch per class
    # - defensive programming
    # - efficient implementation, take advantage of parallelism

    (n, d) = dataset.shape  # number of samples, dimension

    # Determine the sketch dimension and sanity check: the dataset is nonempty and the map works
    if isinstance(featureMap, FeatureMap):  # featureMap is the argument, FeatureMap is the class
        m = featureMap.m
    else:
        try:
            m = featureMap(dataset[0]).shape[0]
        except Exception:
            raise ValueError("Unexpected error while calling the sketch feature map:", sys.exc_info()[0])

    # Split the batches
    # if batch_size is None:
    #     batch_size = int(1e6 / m)  # Rough heuristic, best batch size will vary on different machines
    nb_batches = int(np.ceil(n / batch_size))

    sketch = torch.zeros(m).to(featureMap.device)
    sum_arg = {"dim": 0}

    if datasetWeights is None:
        for b in range(nb_batches):
            if b % 100 == 0 and display:
                logger.info(f"Sketching batch: {b+1}/{nb_batches}")
            sketch = sketch + featureMap(dataset[b * batch_size:(b + 1) * batch_size]).sum(**sum_arg)
        sketch /= n
    else:
        sketch = datasetWeights @ featureMap(dataset)
    return sketch


def get_sketch_Omega_xi_from_aggregation(aggregated_sketch, aggregated_sigmas, directions, aggregated_R, aggregated_xi,
                                         Omega, R_seeds, needed_sigma, needed_seed, keep_all_sigmas, use_torch):
    """

    Parameters
    ----------
    aggregated_sketch:
        The sketch obtained from the "mutualized" sketching operator. The "end-to-end" concatenation of the sketch
        obtained with the various sigmas and seeds.
    aggregated_sigmas:
        The list of possible sigmas ordered in the same order than in the aggregated_sketch.
    directions:
        The base directions, common between the different sketches.
    aggregated_R:
        The list of possible radius samples ordered in the same order than in the aggregated_sketch.
    aggregated_xi:
        The list of possible xi samples ordered in the same order than in the aggregated_sketch.
    Omega:

    R_seeds:
        The list of seeds used to generate the different R ordered in the same order than in the aggregated_sketch.
    needed_sigma:
        The desired sigma. Should be in the list of possible sigmas.
    needed_seed:
        The desired seed. Should be in the list of seeds.
    keep_all_sigmas:
        Tells to keep all the sigmas.
    use_torch
        Return the result as torch.Tensors instead of a numpy.ndarrays.
    Returns
    -------

    """
    # Omega can be equal to None if it is a numpy array containing None (hence not technically None, because a nparray)
    if Omega != None :  # hot potato
        if use_torch:
            return torch.from_numpy(aggregated_sketch), torch.from_numpy(Omega), torch.from_numpy(aggregated_xi)
        else:
            return aggregated_sketch, Omega, aggregated_xi

    sketch_size = len(aggregated_sketch)
    nb_sigmas = len(aggregated_sigmas)
    nb_directions = directions.shape[1]
    # Select the part of the saved sketch corresponding to the wanted seed
    indice_R_seed = np.where(np.array(R_seeds) ==needed_seed)[0][0]
    size_sketch_by_R = sketch_size // len(R_seeds)
    assert sketch_size == len(R_seeds) * size_sketch_by_R
    first_sketch_elm_for_R = size_sketch_by_R * indice_R_seed
    last_sketch_elm_for_R = size_sketch_by_R * (indice_R_seed + 1)

    # update sketch and everything for the rest of the computatation
    aggregated_sketch = aggregated_sketch[first_sketch_elm_for_R:last_sketch_elm_for_R]
    sketch_size = len(aggregated_sketch)
    aggregated_xi = aggregated_xi[first_sketch_elm_for_R:last_sketch_elm_for_R]
    aggregated_R = aggregated_R[:, indice_R_seed]

    if keep_all_sigmas:
        assert (aggregated_sigmas == needed_sigma).all()
    else:
        assert np.isclose(needed_sigma, aggregated_sigmas).any()
        index_needed_sigma = np.where(np.isclose(needed_sigma, aggregated_sigmas))
        assert len(index_needed_sigma[0]) == 1
        index_needed_sigma = index_needed_sigma[0][0]

        assert aggregated_R.shape[0] == nb_directions == sketch_size / nb_sigmas
        first_sketch_elm = index_needed_sigma * nb_directions
        last_sketch_elm = first_sketch_elm + nb_directions

        aggregated_sketch = aggregated_sketch[first_sketch_elm:last_sketch_elm]
        aggregated_xi = aggregated_xi[first_sketch_elm:last_sketch_elm]
        assert len(aggregated_sketch) == nb_directions

    if use_torch:  # cleaning check if torch is necessary
        needed_sigma = torch.from_numpy(needed_sigma) if not is_number(needed_sigma) else torch.Tensor(
            [needed_sigma])
        aggregated_sketch = torch.from_numpy(aggregated_sketch)
        aggregated_xi = torch.from_numpy(aggregated_xi)
        directions = torch.from_numpy(directions)
        aggregated_R = torch.from_numpy(aggregated_R)
        return aggregated_sketch, (needed_sigma, directions, aggregated_R), aggregated_xi
    else:
        return aggregated_sketch, (needed_sigma, directions, aggregated_R), aggregated_xi

### TODOS FOR SKETCHING.PY

# Short-term:
#  - Add support of private sketching for the real variants of the considered maps
#  - Add the square nonlinearity, for sketching for PCA for example

# Long-term:
# - Fast sketch computation