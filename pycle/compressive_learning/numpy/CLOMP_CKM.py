import numpy as np

from pycle.compressive_learning.numpy.CLOMP import CLOMP


##########################
# 1: Compressive K-Means #
##########################
class CLOMP_CKM(CLOMP):
    """
    CLOMP solver for Compressive K-Means (CKM), where we fit a mixture of K Diracs to the sketch.
    The main algorithm is handled by the parent class.
    """

    def __init__(self, Phi, *args, **kwargs):
        super(CLOMP_CKM, self).__init__(phi=Phi, d_atom=Phi.d, *args, **kwargs)

    def sketch_of_atoms(self, theta_k, return_jacobian=False):
        """
        Computes and returns A_Phi(P_theta_k) for an atom P_theta_k.
        possibly with the jacobian, of size (d_atom,m)
        """
        if theta_k.ndim == 1:
            theta_k = theta_k[np.newaxis, :]

        # assert theta_k.size == self.d_atom

        sketch_of_atom = self.phi(theta_k).T

        if return_jacobian:
            jacobian = self.phi.grad(theta_k)
            return sketch_of_atom, jacobian
        else:
            return sketch_of_atom

    def set_bounds_atom(self, bounds):
        """
        Should set self.bounds_atom to a list of length d_atom of lower and upper bounds, i.e.,
            self.bounds_atom = [[lowerbound_1,upperbound_1], ..., [lowerbound_d_atom,upperbound_d_atom]]
        """
        assert bounds.shape == (2, self.phi.d)
        self.bounds = bounds  # data bounds
        self.bounds_atom = bounds.T.tolist()

    def randomly_initialize_new_atom(self):
        new_theta = np.random.uniform(self.bounds[0], self.bounds[1])
        return new_theta

    def get_centroids(self):
        return self.current_sol[1]