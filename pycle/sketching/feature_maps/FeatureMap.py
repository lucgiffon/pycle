from abc import ABC, abstractmethod


class FeatureMap(ABC):
    """Abstract feature map class
    Template for a generic Feature Map. Useful to check if an object is an instance of FeatureMap."""

    def __init__(self, *args, **kwargs):
        self.counter_call_sketching_operator = 0

    @property
    @abstractmethod
    def m(self):
        pass

    def account_call(self, x):
        if len(x.shape) == 1:
            self.counter_call_sketching_operator += 1
        else:
            assert len(x.shape) == 2
            self.counter_call_sketching_operator += x.shape[0]

    def reset_counter(self):
        self.counter_call_sketching_operator = 0

    @abstractmethod
    def call(self, x):
        raise NotImplementedError("The way to compute the feature map is not specified.")

    def __call__(self, x):
        self.account_call(x)
        return self.call(x)

    @abstractmethod
    def grad(self, x):
        raise NotImplementedError("The way to compute the gradient of the feature map is not specified.")