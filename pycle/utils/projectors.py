"""
This module contains projector functions.

These projector function are used in :class:`CLOMP_CKM`, at the end of each iteration,
to let the parameters satisify some constraints. For instance, if the parameters
are wanted to have norm = 1, there is a projector just to do that.

Any custom projector can be implemented from the :class:`Projector` base abstract class.

They are used with the :class:`CLOMP_CKM` like::

    from pycle.utils.projectors import ProjectorClip

    my_proj = ProjectorClip(torch.tensor(-1), torch.tensor(-1))
    CLOMP_CKM(phi=Phi, size_mixture_K=nb_clust, bounds=bounds, sketch_z=z, projector=my_proj)

To build a custom projector, create a class that inherits from the :class:`Projector` class and override the
`project` method tomake inplace modification of the parameters in the optimization. For example::

    class ProjectorClip(Projector):

        def __init__(self, lower_bound: torch.Tensor, upper_bound: torch.Tensor):
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

        def project(self, param):
            torch.minimum(param, self.upper_bound, out=param)  # modifications are made in place
            torch.maximum(param, self.lower_bound, out=param)


"""

from abc import ABCMeta, abstractmethod
from typing import NoReturn

import torch


class Projector(metaclass=ABCMeta):
    """
    Base, asbtract, class for projectors which only need to implement the :meth:`project` method.
    """
    @abstractmethod
    def project(self, param) -> NoReturn:
        """
        This function will take the parameters as input and make the projection on it.

        The projection must happen **in place**. It means that the `project` method doesn't
        return anything but instead modify the object identified by `param` directly.

        Parameters
        ----------
        param:
            The parameters to project.


        """
        raise NotImplementedError


class ProjectorNoProjection(Projector):
    """
    Dummy, place handler, projector class to do nothing.
    """
    def project(self, param):
        pass


class ProjectorClip(Projector):
    """
    Clip the input parameters to lower and upper bounds.
    """
    def __init__(self, lower_bound: torch.Tensor, upper_bound: torch.Tensor):
        """

        Parameters
        ----------
        lower_bound:
            The minimum acceptable value for the parameters.
        upper_bound
            The maximum acceptable value for the parameters.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def project(self, param):
        assert isinstance(param, torch.Tensor)
        self.upper_bound = self.upper_bound.to(param)

        torch.minimum(param, self.upper_bound, out=param)
        torch.maximum(param, self.lower_bound, out=param)


class ProjectorLessUnit2Norm(Projector):
    """
    Clip the input row vectors to norm 1:

    the row vectors with norm greater than 1 are clipped to have norm equal to 1.
    Other row vectors are left untouched.
    """
    def project(self, param):
        norm = torch.norm(param, dim=-1)
        if len(param.shape) == 2:
            indices = norm > 1
            if torch.any(indices):
                param[indices] = param[indices] / norm[indices].unsqueeze(-1)
        else:
            if norm.item() > 1:
                torch.div(param, norm, out=param)


class ProjectorExactUnit2Norm(Projector):
    """
    Normalize the row vectors to 1 (divide by their respective norm).
    """
    def project(self, param):
        norm = torch.norm(param, dim=-1)
        torch.div(param, norm.unsqueeze(-1), out=param)
