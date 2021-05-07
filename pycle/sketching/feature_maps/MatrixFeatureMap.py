import numpy as np

from pycle.sketching.feature_maps.SimpleFeatureMap import SimpleFeatureMap


# schellekensvTODO find a better name
class MatrixFeatureMap(SimpleFeatureMap):
    """Feature map the type Phi(x) = c_norm*f(Omega^T*x + xi)."""

    def __init__(self, f, Omega, **kwargs):
        # 2) extract Omega the projection matrix schellekensvTODO allow callable Omega for fast transform
        self.Omega = Omega
        super().__init__(f, **kwargs)

    def init_shape(self):
        try:
            return self.Omega.shape
        except AttributeError:
            raise ValueError("The provided projection matrix Omega should be a (d,m) linear operator.")

    # call the FeatureMap object as a function
    def call(self, x):
        # return self.c_norm*self.f(np.matmul(self.Omega.T,x.T).T + self.xi) # Evaluate the feature map at x
        return self.c_norm * self.f(np.matmul(x, self.Omega) + self.xi)  # Evaluate the feature map at x

    def grad(self, x):
        """Gradient (Jacobian matrix) of Phi, as a (d,m)-numpy array"""
        return self.c_norm * self.f_grad(np.matmul(self.Omega.T, x.T).T + self.xi) * self.Omega
