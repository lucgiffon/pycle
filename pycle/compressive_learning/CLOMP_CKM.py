"""
Contains the CLOMP_CKM class to solve the compressive kmeans problem.
"""
from typing import NoReturn

from pycle.compressive_learning.CLOMP import CLOMP
from pycle.sketching import FeatureMap
from pycle.utils.projectors import Projector, ProjectorClip, ProjectorNoProjection
import torch
import numpy


class CLOMP_CKM(CLOMP):
    """
    Instanciate a CLOMP solver for the specific case of compressive kmeans. That is compressive learning of centroids.

    Cluster center and centroid are interchangeable words for the same concept. They are the location of each dirac
    representing one mixture component.

    The definitions are specialized, more precise, less abstract, now:

    - The sketched component k of the mixture model is denoted phi_theta_k. It is the feature map applied to a centroid.
    - The parameters of the component k of the mixture model is denoted theta_k. It is a centroid.
    - The dimension of each parameter vector is D. It is the dimension of the underlying data.
    - The size of the mixture is K, the number of components. It is also the number of centroids/clusters.
    - The alphas are the weights of the mixture. It correspond to the weights, the importance of each centroid.
    - The solution is the pair (alphas, thetas), that is, all the parameters of the mixture
    """
    def __init__(self, phi: FeatureMap, centroid_projector: Projector = ProjectorNoProjection(), *args, **kwargs):
        """
        Parameters
        ----------
        phi
            The feature map used in the sketching operator.
        centroid_projector:
            A callback projector object to call on the centroids at the end of each iteration
            in order to enforce some constrains.
        args
        kwargs
        """

        # Lower and upper bounds are for random initialization
        self.lower_bounds = None
        self.upper_bounds = None

        super().__init__(phi, thetas_dimension_D=phi.d, *args, **kwargs)

        assert isinstance(centroid_projector, Projector)
        self.centroid_projector = centroid_projector
        if isinstance(self.centroid_projector, ProjectorClip):
            # note that bounds for projection and bounds for initialization are different entities
            self.centroid_projector.lower_bound = self.centroid_projector.lower_bound.to(self.real_dtype).to(self.device)
            self.centroid_projector.upper_bound = self.centroid_projector.upper_bound.to(self.real_dtype).to(self.device)

    def sketch_of_mixture_components(self, thetas: torch.Tensor):
        """
        Computes and returns phi(theta_k) for each centroid in theta.

        D is the dimension of centroid, M is the dimension of a sketch.

        Parameters
        ----------
        thetas
            (D,) or (current_size_mixture,D)-shaped tensor containing centroids.

        Returns
        -------
            (M,) or (current_size_mixture, M)-shaped tensor constaining the M-dimensional feature maps of the centroids,
            e.g. the atoms.
        """
        assert thetas.size()[-1] == self.thetas_dimension_D
        return self.phi(thetas)

    def set_bounds_thetas(self, bounds: torch.Tensor) -> NoReturn:
        """
        Set the bounds where the centroids can be found.

        These bounds can be used for initizalizing new centroids
        and for setting bounds to the optimization procedure.

        Parameters
        ----------
        bounds
            (2, D)- shaped tensor containing the lower bounds in position 0 and upper bounds in position 1.
        """
        assert len(bounds) == 2
        self.lower_bounds = bounds[0].to(self.real_dtype).to(self.device)
        self.upper_bounds = bounds[1].to(self.real_dtype).to(self.device)
        self.bounds = bounds  # data bounds
        # self.bounds_atom =
        # [[lowerbound_1, upperbound_1],
        #  ...,
        #  [lowerbound_d_atom, upperbound_d_atom]]
        self.bounds_atom = bounds.T.tolist()

    def randomly_initialize_several_mixture_components(self, nb_mixture_components: int):
        """
        Uniform initialization of several centroids between the lower and upper bounds.

        Parameters
        ----------
        nb_mixture_components
            The number of centroids to initialize.

        Returns
        -------
            (nb_atoms, D) shaped tensor containing the centroids.
        """
        all_new_theta = (self.upper_bounds -
                         self.lower_bounds) * torch.rand(nb_mixture_components, self.thetas_dimension_D).to(self.device) + self.lower_bounds
        return all_new_theta

    def projection_step(self, thetas: torch.Tensor) -> NoReturn:
        """
        Project a centroid theta (or a set of thetas) on the constraint specifed
        by self.centroid_project of class `Projector`.

        The modification is made in place.

        Parameters
        ----------
        thetas
            The centroids to project
        """
        if self.centroid_projector is not None:
            self.centroid_projector.project(thetas)

    def get_centroids(self, return_numpy=False) -> [torch.Tensor, numpy.ndarray]:
        """
        Return the centroids.

        Parameters
        ----------
        return_numpy
            If True, return as numpy array (default: False).

        Returns
        -------
            (current_size_mixture, D) shaped tensor or ndarray containing the centroids.
        """
        if return_numpy:
            return self.thetas.cpu().numpy()
        return self.thetas
