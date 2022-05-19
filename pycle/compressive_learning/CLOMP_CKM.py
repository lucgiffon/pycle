from typing import NoReturn

from pycle.compressive_learning.CLOMP import CLOMP
from pycle.sketching import FeatureMap
from pycle.utils.projectors import Projector, ProjectorClip, ProjectorNoProjection
import torch
import numpy


# cleaning use standardized naming for cluster centers and atoms
#  think well what each solver is looking for.
#  define the notions of: cluster centers/theta/centroids/atoms/all_thetas/Theta/mixture component
class CLOMP_CKM(CLOMP):
    """
    Instanciate a CLOMP solver specific of compressive kmeans.
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

        super().__init__(phi, D_theta=phi.d, *args, **kwargs)

        assert isinstance(centroid_projector, Projector)
        self.centroid_projector = centroid_projector
        if isinstance(self.centroid_projector, ProjectorClip):
            # note that bounds for projection and bounds for initialization are different entities
            self.centroid_projector.lower_bound = self.centroid_projector.lower_bound.to(self.real_dtype).to(self.device)
            self.centroid_projector.upper_bound = self.centroid_projector.upper_bound.to(self.real_dtype).to(self.device)

    def sketch_of_atoms(self, theta: torch.Tensor):
        """
        Computes and returns A_Phi(theta_k) for each cluster center in theta.

        D is the dimension of cluster center, M is the dimension of a sketch.

        Parameters
        ----------
        theta
            (D,) or (n_atoms,D)-shaped tensor containing cluster centers.

        Returns
        -------
            (M,) or (n_atoms, M)-shaped tensor constaining the M-dimensional feature maps of the cluster centers,
            e.g. the atoms.
        """
        assert theta.size()[-1] == self.d_theta
        return self.phi(theta)

    def set_bounds_atom(self, bounds: torch.Tensor) -> NoReturn:
        """
        Set the bounds where the cluster centers can be found.

        These bounds can be used for initizalizing new cluster centers
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

    def randomly_initialize_several_atoms(self, nb_atoms: int):
        """
        Uniform initialization of several cluster centers between the lower and upper bounds.

        Parameters
        ----------
        nb_atoms
            The number of cluster centers to initialize.

        Returns
        -------
            (nb_atoms, D) shaped tensor containing the cluster centers.
        """
        all_new_theta = (self.upper_bounds -
                         self.lower_bounds) * torch.rand(nb_atoms, self.d_theta).to(self.device) + self.lower_bounds
        return all_new_theta

    def projection_step(self, theta: torch.Tensor) -> NoReturn:
        """
        Project a cluster center theta (or a set of thetas) on the constraint specifed
        by self.centroid_project of class `Projector`.

        The modification is made in place.

        Parameters
        ----------
        theta
        """
        if self.centroid_projector is not None:
            self.centroid_projector.project(theta)

    def get_centroids(self, return_numpy=False) -> [torch.Tensor, numpy.ndarray]:
        """
        Return the cluster centers.

        Parameters
        ----------
        return_numpy
            If True, return as numpy array (default: False).

        Returns
        -------
            (n_atoms, D) shaped tensor or ndarray containing the cluster centers.
        """
        if return_numpy:
            return self.all_thetas.cpu().numpy()
        return self.all_thetas

