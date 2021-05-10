from pycle.sketching.feature_maps.SimpleFeatureMap import SimpleFeatureMap
from lightonml import OPU
from lightonml.encoding.base import SeparatedBitPlanEncoder, MixingBitPlanDecoder
from lightonml.internal.simulated_device import SimulatedOpuDevice
import numpy as np

# schellekensvTODO find a better name
class OPUFeatureMap(SimpleFeatureMap):
    """Feature map the type Phi(x) = c_norm*f(OPU(x) + xi)."""

    def __init__(self, f, opu, Sigma=None, **kwargs):
        # 2) extract Omega the projection matrix schellekensvTODO allow callable Omega for fast transform
        # todoopu initialiser l'opu
        self.encoder = SeparatedBitPlanEncoder()
        self.decoder = MixingBitPlanDecoder()
        self.opu = opu

        super().__init__(f, **kwargs)

        # todoopu deplacer ca
        self.Sigma = Sigma
        if self.Sigma is None:
            self.Sigma = np.identity(self.opu.max_n_features) # todoopu attention a ca
        self.SigFact = np.linalg.inv(np.linalg.cholesky(self.Sigma))
        self.R = np.abs(np.random.randn(self.m))  # folded standard normal distribution radii
        self.norm_scaling = 1. / np.sqrt(self.d) * np.ones(self.m)

    def init_shape(self):
        if isinstance(self.opu.device, SimulatedOpuDevice):
            return self.opu.max_n_features, self.opu.n_components
        else:
            return None, self.opu.n_components

    def applyOPU(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x = x @ self.SigFact

        x_enc = self.encoder.transform(x)
        y_enc = self.opu.fit_transform1d(x_enc)
        y_dec = self.decoder.transform(y_enc)

        y_dec_expected = np.real(self.opu.device.random_matrix.T @ x.T).T
        y_dec_expected2 = (np.real(self.opu.device.random_matrix.T) @ x.T).T
        y_dec_complex = (self.opu.device.random_matrix.T @ x.T).T
        raise NotImplementedError("OPU feature map is not ready to use yet.")
        # todo fix this

        out = self.R * y_dec * self.norm_scaling
        # Om = SigFact @ phi * R
        return out

    # call the FeatureMap object as a function
    def call(self, x):
        # return self.c_norm*self.f(np.matmul(self.Omega.T,x.T).T + self.xi) # Evaluate the feature map at x
        return self.c_norm * self.f(self.applyOPU(x) + self.xi)  # Evaluate the feature map at x

    def grad(self, x):
        raise NotImplementedError("OPU doesn't have a gradient.")