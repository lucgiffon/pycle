import numpy as np

from pycle.sketching.feature_maps import _dico_nonlinearities
from pycle.sketching.feature_maps.FeatureMap import FeatureMap
from scipy.sparse.linalg import aslinearoperator

# schellekensvTODO find a better name
class SimpleFeatureMap(FeatureMap):
    """Feature map the type Phi(x) = c_norm*f(Omega^T*x + xi)."""

    def __init__(self, f, Omega, xi=None, c_norm=1.):
        """
        - f can be one of the following:
            -- a string for one of the predefined feature maps:
                -- "complexExponential"
                -- "universalQuantization"
                -- "cosine"
            -- a callable function
            -- a tuple of function (specify the derivative too)

        """
        # 1) extract the feature map
        self.name = None
        if isinstance(f, str):
            try:
                (self.f, self.f_grad) = _dico_nonlinearities[f.lower()]
                self.name = f  # Keep the feature function name in memory so that we know we have a specific fct
            except KeyError:
                raise NotImplementedError("The provided feature map name f is not implemented.")
        elif callable(f):
            (self.f, self.f_grad) = (f, None)
        elif (isinstance(f, tuple)) and (len(f) == 2) and (callable(f[0]) and callable(f[1])):
            (self.f, self.f_grad) = f
        else:
            raise ValueError("The provided feature map f does not match any of the supported types.")

        # 2) extract Omega the projection matrix schellekensvTODO allow callable Omega for fast transform
        try:
            self.Omega = Omega
            (self.d, self._m) = self.Omega.shape
        except TypeError:
            raise ValueError("The provided projection matrix Omega should be a (d,m) linear operator.")

        # 3) extract the dithering
        if xi is None:
            self.xi = np.zeros(self._m)
        else:
            self.xi = xi
        # 4) extract the normalization constant
        if isinstance(c_norm, str):
            if c_norm.lower() in ['unit', 'normalized']:
                self.c_norm = 1. / np.sqrt(self._m)
            else:
                raise NotImplementedError("The provided c_norm name is not implemented.")
        else:
            self.c_norm = c_norm

        super().__init__()

    @property
    def m(self):
        return self._m

    # call the FeatureMap object as a function
    def call(self, x):
        # return self.c_norm*self.f(np.matmul(self.Omega.T,x.T).T + self.xi) # Evaluate the feature map at x
        return self.c_norm * self.f(np.matmul(x, self.Omega) + self.xi)  # Evaluate the feature map at x

    def grad(self, x):
        """Gradient (Jacobian matrix) of Phi, as a (d,m)-numpy array"""
        return self.c_norm * self.f_grad(np.matmul(self.Omega.T, x.T).T + self.xi) * self.Omega