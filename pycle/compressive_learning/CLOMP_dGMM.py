import numpy as np

from pycle.compressive_learning import CLOMP
from pycle.sketching import SimpleFeatureMap, fourierSketchOfGaussian, estimate_Sigma_from_sketch


########################
#  2: Compressive GMM  #
########################
## 2.1 (diagonal) GMM with CLOMP
class CLOMP_dGMM(CLOMP):
    """
    CLOMP solver for diagonal Gaussian Mixture Modeling (dGMM), where we fit a mixture of K Gaussians
    with diagonal covariances to the sketch.
    The main algorithm is handled by the parent class.
    Requires the feature map to be Fourier features.

    Init_variance_mode is either "bounds" or "sketch" (default).
    """

    def __init__(self, Phi, K, bounds, sketch=None, sketch_weight=1., init_variance_mode="sketch", verbose=0):
        # Check that the feature map is an instance of RFF, otherwise computations are wrong
        assert isinstance(Phi, SimpleFeatureMap)
        assert Phi.name.lower() == "complexexponential"
        self.bounds_atom = None

        self.variance_relative_lowerbound = (
                                                1e-4) ** 2  # Lower bound on the variances, relative to the data domain size
        self.variance_relative_upperbound = 0.5 ** 2  # Upper bound on the variances, relative to the data domain size

        d_atom = 2 * Phi.d  # d parameters for the Gaussian centers and d parameters for the diagonal covariance matrix
        super(CLOMP_dGMM, self).__init__(Phi, K, d_atom, bounds, sketch, sketch_weight, verbose)

        self.init_variance_mode = init_variance_mode

    def sketch_of_atom(self, theta_k, return_jacobian=False):
        """
        Computes and returns A_Phi(P_theta_k) for an atom P_theta_k.
        possibly with the jacobian, of size (d_atom,m)
        """
        assert theta_k.size == self.d_atom

        (mu, sig) = (theta_k[:self.Phi.d], theta_k[-self.Phi.d:])
        sketch_of_atom = fourierSketchOfGaussian(mu, np.diag(sig), self.Phi.Omega, self.Phi.xi, self.Phi.c_norm)

        if return_jacobian:
            jacobian = 1j * np.zeros((self.d_atom, self.Phi.m))
            jacobian[:self.Phi.d] = 1j * self.Phi.Omega * sketch_of_atom  # Jacobian w.r.t. mu
            jacobian[self.Phi.d:] = -0.5 * (self.Phi.Omega ** 2) * sketch_of_atom  # Jacobian w.r.t. sigma
            return sketch_of_atom, jacobian
        else:
            return sketch_of_atom

    def set_bounds_atom(self, bounds):
        """
        Should set self.bounds_atom to a list of length d_atom of lower and upper bounds, i.e.,
            self.bounds_atom = [[lowerbound_1,upperbound_1], ..., [lowerbound_d_atom,upperbound_d_atom]]
        """
        assert bounds.shape == (2, self.Phi.d)
        self.bounds = bounds  # data bounds
        self.bounds_atom = bounds.T.tolist()
        for i in range(self.Phi.d):  # bounds for the variance in each dimension
            max_variance_this_dimension = (bounds[1][i] - bounds[0][i]) ** 2
            self.bounds_atom.append([self.variance_relative_lowerbound * max_variance_this_dimension,
                                     self.variance_relative_upperbound * max_variance_this_dimension])

    def randomly_initialize_new_atom(self):
        mu0 = np.random.uniform(self.bounds[0], self.bounds[1])  # initial mean
        # check we can use sketch heuristic (large enough m)
        MINIMAL_C_VALUE = 6
        MAXIMAL_C_VALUE = 25
        MINIMAL_POINTS_PER_BOX = 5
        if self.init_variance_mode == "sketch":
            c = max(self.Phi.m // MINIMAL_POINTS_PER_BOX, MAXIMAL_C_VALUE)
            if c < MINIMAL_C_VALUE:
                self.init_variance_mode = "bounds"

        if self.init_variance_mode == "sketch":
            sigma2_bar = estimate_Sigma_from_sketch(self.sketch, self.Phi, c=c)
            sig0 = sigma2_bar[0] * np.ones(self.Phi.d)
        elif self.init_variance_mode == "bounds":
            sig0 = (10 ** np.random.uniform(-0.8, -0.1, self.Phi.d) * (
                    self.bounds[1] - self.bounds[0])) ** 2  # initial covariances
        else:
            raise NotImplementedError

        new_theta = np.append(mu0, sig0)
        return new_theta

    def get_GMM(self):
        (weights, _Theta) = self.current_sol
        (K, d) = self.n_atoms, self.Phi.d
        mus = np.zeros((K, d))
        Sigmas = np.zeros((K, d, d))
        for k in range(K):
            mus[k] = _Theta[k][:d]
            Sigmas[k] = np.diag(_Theta[k][d:])

        return weights, mus, Sigmas