from pycle.compressive_learning.CLOMP import CLOMP
from pycle.sketching import FeatureMap
from pycle.utils.projectors import Projector, ProjectorClip, ProjectorNoProjection
import torch
import numpy


# cleaning see what is the point of this class and make test
class CLOMP_DL(CLOMP):
    def __init__(self, phi: FeatureMap, dictionary: [torch.Tensor], centroid_projector: Projector = ProjectorNoProjection(), *args, **kwargs):

        # Lower and upper bounds are for random initialization, not for projection step !
        self.lower_bounds = None
        self.upper_bounds = None

        super().__init__(phi, d_theta=phi.d, *args, **kwargs)

        self.dictionary = dictionary

        assert isinstance(centroid_projector, Projector)
        self.centroid_projector = centroid_projector
        if isinstance(self.centroid_projector, ProjectorClip):
            self.centroid_projector.lower_bound = self.centroid_projector.lower_bound.to(self.real_dtype).to(self.device)
            self.centroid_projector.upper_bound = self.centroid_projector.upper_bound.to(self.real_dtype).to(self.device)

    def regularize(self, thetas):
        assert thetas.ndim == 2
        return self.lambda_l1 * torch.linalg.norm(thetas, ord=1, dim=1)

    def loss_atom_correlation(self, theta):
        return super().loss_atom_correlation(theta) + self.regularize(theta)

    def loss_global(self, alphas, all_thetas=None, all_atoms=None):
        return super().loss_global(alphas, all_thetas, all_atoms) + self.regularize(all_thetas)

    def sketch_of_atoms(self, theta):
        """
        Computes and returns A_Phi(P_theta_k) for one or several atoms P_theta_k.
        d is the dimension of atom, m is the dimension of sketch
        :param theta: tensor of size (d) or (K, d)
        :return: tensor of size (m) or (K, m)
        """
        assert theta.size()[-1] == self.d_theta
        return self.phi(theta @ self.dictionary)

    def set_bounds_atom(self, bounds):
        """
        Should set self.bounds_atom to a list of length d_atom of lower and upper bounds, i.e.,
            self.bounds_atom = [[lowerbound_1,upperbound_1], ..., [lowerbound_d_atom,upperbound_d_atom]]
        """
        assert len(bounds) == 2
        self.lower_bounds = bounds[0].to(self.real_dtype).to(self.device)
        self.upper_bounds = bounds[1].to(self.real_dtype).to(self.device)
        self.bounds = bounds  # data bounds
        self.bounds_atom = bounds.T.tolist()

    def randomly_initialize_several_atoms(self, nb_atoms):
        """
        Uniform initialization of several centroids between the lower and upper bounds.
        :return: tensor
        """
        all_new_theta = (self.upper_bounds -
                         self.lower_bounds) * torch.rand(nb_atoms, self.d_theta).to(self.device) + self.lower_bounds
        return all_new_theta

    def projection_step(self, theta) -> None:
        """
        Project a theta (or a set of thetas) on the constraint specifed by self.centroid_project of class `Projector`.

        The modification is made in place.

        :param theta:
        :return: None
        """
        if self.centroid_projector is not None:
            # todo peut etre faire une projection en plus pour ammener le résultat plus près du résultat voulu
            #  en terme de nombre de coefficients
            self.centroid_projector.project(theta)

    def get_centroids(self, return_numpy=True) -> [torch.Tensor, numpy.ndarray]:
        """
        Get centroids, in numpy array by default, otherwise in torch.Tensor.

        :param return_numpy: bool
        :return: tensor or numpy array
        """
        if return_numpy:
            return self.all_thetas.cpu().numpy()
        return self.all_thetas

