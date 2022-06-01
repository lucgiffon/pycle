"""
Sketching functions and modules.
"""
from typing import Callable, Union, Any, Optional

import sys

import numpy as np
import torch

from loguru import logger

from pycle.sketching.feature_maps.FeatureMap import FeatureMap
from pycle.sketching.feature_maps.MatrixFeatureMap import MatrixFeatureMap
from pycle.sketching.frequency_sampling import drawFrequencies
from pycle.utils import is_number


def computeSketch(dataset: torch.Tensor, featureMap: Union[Callable, FeatureMap], datasetWeights: torch.Tensor = None,
                  batch_size: int = 100, verbose: bool = True) -> torch.Tensor:
    """
    Computes the sketch of a dataset given a generic feature map.

    More precisely, evaluates:

    .. math::
        z = \\frac{1}{N} \\sum_{i=1}^{N} \\phi(\\bm{x}_i) * w_i

    where :math:`X` is the dataset, :math:`\\phi` is the sketch feature map, :math:`w_i` are weights assigned to the
    samples (typically 1/N).

    Arguments
    ---------
    dataset
        (N, D)-shaped torch Tensor, the dataset X: N examples in dimension D
    featureMap
        The feature map Phi, given as one of the following:

        - a function: taking a (D,)- and returning (M,)- shaped torch Tensors
        - a :class:`pycle.sketching.feature_maps.FeatureMap.FeatureMap` instance\
        (example: :class:`pycle.sketching.feature_maps.MatrixFeatureMap.MatrixFeatureMap`)
    datasetWeights
        (N,) torch tensor, optional weigths w_i in the sketch (default: None, corresponds to w_i = 1/N)
    batch_size:
        The sketch is computed by chunks of size ``batch_size``.
    verbose:
        If True, logs the current batch number every 100 batches.

    Returns
    -------
    (M,)-shaped torch.Tensor
        The sketch of the dataset by the featureMap.
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
            if b % 100 == 0 and verbose:
                logger.info(f"Sketching batch: {b+1}/{nb_batches}")
            sketch = sketch + featureMap(dataset[b * batch_size:(b + 1) * batch_size]).sum(**sum_arg)
        sketch /= n
    else:
        sketch = datasetWeights @ featureMap(dataset)
    return sketch


def get_sketch_Omega_xi_from_aggregation(aggregated_sketch: np.ndarray, aggregated_sigmas, directions,
                                         aggregated_R, aggregated_xi,
                                         Omega, R_seeds, needed_sigma, needed_seed, keep_all_sigmas, use_torch=True):
    """
    Note that this function works with numpy arrays as input and not torch tensors.

    From an aggregated sketch, get the sub-sketch of interest.

    "Mutualized" sketching is sharing the matrix of directions between the computation of many sketches with
    many scaling factors or many seeds for the amplitude sampling R.
    The vector obtained at the outcome of the mutualized sketching is the "aggregated sketch".
    One may want to retrieve the sketch corresponding to one particular scaling factor sigma or seed.
    This function allows to do so.

    The aggregated sketch is constructed in this order::

        sketch = []
        for R_seed in lst_R_seeds:
            for sigma in lst_sigmas:
                sub_sketch = sketching(seed, sigma)
                sketch = sketch.concat_to_the_end(sub_sketch)

    This means that the m-sized sketch corresponding to the i_th R_seed and the j_th sigma value is located at::

        start = i * (m * len(lst_sigmas)) + j * m
        end = start + m

    i and j starts at 0.

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
        If Omega is provided (different than None), it means the aggregated sketch, Omega and xi must be returned as is.
    R_seeds:
        The list of seeds used to generate the different R ordered in the same order than in the aggregated_sketch.
    needed_sigma:
        The desired sigma. Should be in the list of possible sigmas.
    needed_seed:
        The desired seed. Should be in the list of seeds.
    keep_all_sigmas:
        Tells to keep all the sigmas so the returned sketch will be an aggregation of the sketches obtained with
        the different sigmas.
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

    if use_torch:
        needed_sigma = torch.from_numpy(needed_sigma) if not is_number(needed_sigma) else torch.Tensor(
            [needed_sigma])
        aggregated_sketch = torch.from_numpy(aggregated_sketch)
        aggregated_xi = torch.from_numpy(aggregated_xi)
        directions = torch.from_numpy(directions)
        aggregated_R = torch.from_numpy(aggregated_R)
        return aggregated_sketch, (needed_sigma, directions, aggregated_R), aggregated_xi
    else:
        return aggregated_sketch, (needed_sigma, directions, aggregated_R), aggregated_xi


def pick_sketch_by_indice_in_aggregated_sketch(aggregated_sketch: np.ndarray, index_needed_sigma: int, sketch_size: int,
                                               indice_R_seed: Optional[int] = None,
                                               number_R_seeds: Optional[int] = None) -> np.ndarray:
    """
    From a mutualized sketch, get the sub-sketch of interest.

    "Mutualized" sketching is sharing the matrix of directions between the computation of many sketches with
    many scaling factors or many seeds for the amplitude sampling R.
    The vector obtained at the outcome of the mutualized sketching is the "aggregated sketch".
    One may want to retrieve the sketch corresponding to one particular scaling factor sigma or seed.
    This function allows to do so.

    The aggregated sketch is constructed in this order::

        sketch = []
        for sigma in lst_sigmas:
            sub_sketch = sketching(seed, sigma)
            sketch = sketch.concat_to_the_end(sub_sketch)

    This means that the m-sized sketch corresponding to the i_th R_seed and the j_th sigma value is located at::

        start = i * (m * len(lst_sigmas)) + j * m
        end = start + m

    i and j starts at 0.

    Parameters
    ----------
    aggregated_sketch
        The sketch obtained from the "mutualized" sketching operator. The "end-to-end" concatenation of the sketch
        obtained with the various sigmas and seeds.
    index_needed_sigma
        The desired indice in the list of sigmas. It is the position of the sketch in the aggregated sketch.
    sketch_size
        The size of a single sub-sketch in the mutualized sketch.
    indice_R_seed
        The indice of the sampling of R if there is more than one.
    number_R_seeds
        The number of sampling of R.

    Notes
    -----
    - `indice_R_seed` and `number_R_seeds` must both be set or None
    - This function works with numpy arrays as input/output and not torch tensors.

    Returns
    -------
        The subsketch of interest
    """

    mutualized_sketch_size = len(aggregated_sketch)

    # first choose the boundaries of the sketch for a given seed (if seed is a concern)
    assert (indice_R_seed is not None and number_R_seeds is not None) or (indice_R_seed is None and number_R_seeds is None), \
        "None or both indice_R_seed and number_R_seeds parameters must be provided"
    if number_R_seeds is not None:
        # Select the part of the saved sketch corresponding to the wanted seed
        size_sketch_by_R = mutualized_sketch_size // number_R_seeds
        assert mutualized_sketch_size == number_R_seeds * size_sketch_by_R
        first_sketch_elm_for_R = size_sketch_by_R * indice_R_seed
        last_sketch_elm_for_R = size_sketch_by_R * (indice_R_seed + 1)
    else:
        first_sketch_elm_for_R = 0
        last_sketch_elm_for_R = mutualized_sketch_size

    # update sketch and everything for the rest of the computatation
    aggregated_sketch = aggregated_sketch[first_sketch_elm_for_R:last_sketch_elm_for_R]

    first_sketch_elm = index_needed_sigma * sketch_size
    last_sketch_elm = first_sketch_elm + sketch_size

    aggregated_sketch = aggregated_sketch[first_sketch_elm:last_sketch_elm]

    assert len(aggregated_sketch) == sketch_size

    return aggregated_sketch



