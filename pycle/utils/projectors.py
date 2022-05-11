from abc import ABCMeta, abstractmethod
import torch


class Projector(metaclass=ABCMeta):
    """
    Base, asbtract, class for projectors which only need to implement the `projet` method.
    """
    @abstractmethod
    def project(self, param):
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
    def __init__(self, lower_bound, upper_bound):
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
