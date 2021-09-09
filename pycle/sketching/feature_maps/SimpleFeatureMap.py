import numpy as np

from pycle.sketching.feature_maps import _dico_nonlinearities
from pycle.sketching.feature_maps.FeatureMap import FeatureMap
from abc import abstractmethod, ABCMeta


# schellekensvTODO find a better name
class SimpleFeatureMap(FeatureMap, metaclass=ABCMeta):
    def __init__(self, f, xi=None, c_norm=1., encoding_decoding=False):
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

        self.d, self._m = self.init_shape()

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

        self.encoding_decoding = encoding_decoding

        super().__init__()

    @abstractmethod
    def init_shape(self):
        pass

    @property
    def m(self):
        return self._m

    @abstractmethod
    def call(self, x):
        pass

    @abstractmethod
    def grad(self, x):
        pass
