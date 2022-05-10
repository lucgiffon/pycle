from abc import ABCMeta, abstractmethod
import torch


# cleaning check if these are useful
class Projector(metaclass=ABCMeta):
    @abstractmethod
    def project(self, param):
        raise NotImplementedError


class ProjectorNoProjection(Projector):
    def project(self, param):
        pass


class ProjectorClip(Projector):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def project(self, param):
        assert isinstance(param, torch.Tensor)
        self.upper_bound = self.upper_bound.to(param)

        torch.minimum(param, self.upper_bound, out=param)
        torch.maximum(param, self.lower_bound, out=param)


class ProjectorLessUnit2Norm(Projector):
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
    def project(self, param):
        norm = torch.norm(param, dim=-1)
        torch.div(param, norm.unsqueeze(-1), out=param)
