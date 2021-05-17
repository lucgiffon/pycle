from pycle.sketching.feature_maps.SimpleFeatureMap import SimpleFeatureMap
from lightonml import OPU
from lightonml.encoding.base import SeparatedBitPlanEncoder, MixingBitPlanDecoder
from lightonml.internal.simulated_device import SimulatedOpuDevice
import numpy as np
from scipy.linalg import hadamard
from fht import fht

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

        self.mu_opu, self.std_opu = self.get_distribution_opu()
        self.opu.fit1d(n_features=self.d)

        # todoopu deplacer ca
        self.Sigma = Sigma
        if self.Sigma is None:
            self.Sigma = np.identity(self.opu.max_n_features)  # todoopu attention a ca car ce n'est disponible qu'en mode simulé
        self.SigFact = np.linalg.inv(np.linalg.cholesky(self.Sigma))
        self.R = np.abs(np.random.randn(self.m))  # folded standard normal distribution radii
        self.norm_scaling = 1. / np.sqrt(self.d) * np.ones(self.m)

    def get_distribution_opu(self, light_memory=False):
        H = hadamard(self.d)
        B = np.array((self.opu.transform(H > 0) - self.opu.transform(H < 0)))
        # B = transform_batch(H, opu).T

        sqrt_d = np.sqrt(self.d)
        # HB = np.array(H @B)
        # O = 1./in_dim * HB
        if not light_memory:
            FHB = np.array([1./self.d * fht(b) * sqrt_d for b in B.T]).T
            mu = np.mean(FHB)
            std = np.std(FHB)

        else:
            sum_mu = 0
            count = 0
            for b in B.T:
                col = 1./self.d * fht(b) * sqrt_d
                sum_mu += np.sum(col)
                count += col.size
            mu = sum_mu / count
            sum_var = 0
            for b in B.T:
                col = 1./self.d * fht(b) * sqrt_d
                sum_var += np.sum((mu - col)**2)
            var = sum_var / count
            std = np.sqrt(var)

        return mu, std

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
        # todo verifier que les plans de bits sont biens en axis=0
        y_enc = self.opu.transform(x_enc)
        # now center the coefficients of the matrix
        mu_x_enc = self.mu_opu * np.sum(x_enc, axis=1)  # sum other all dims
        y_enc = y_enc - mu_x_enc.reshape(-1, 1)

        # now scale the result
        y_enc = self.R * y_enc * 1./(self.std_opu**2) * self.norm_scaling
        y_dec = self.decoder.transform(y_enc)

        # now center the coefficients of the matrix
        # mu_x = self.mu_opu * np.sum(x, axis=1)  # sum other all dims
        # y_dec = y_dec - mu_x.reshape(-1, 1)
        #
        # now scale the result
        # y_dec = y_dec * 1./(self.std_opu**2)

        out = y_dec
        # Om = SigFact @ phi * R
        return out

    # call the FeatureMap object as a function
    def call(self, x):
        # return self.c_norm*self.f(np.matmul(self.Omega.T,x.T).T + self.xi) # Evaluate the feature map at x
        return self.c_norm * self.f(self.applyOPU(x) + self.xi)  # Evaluate the feature map at x

    def grad(self, x):
        raise NotImplementedError("OPU doesn't have a gradient.")