from abc import abstractmethod, ABCMeta
import numpy as np

class Kernel(metaclass=ABCMeta):
    def __init__(self,x1,x2, *hparam, **kwargs):
        self.x1 = self.toArray(*x1)
        self.x2 = self.toArray(*x2)

    @abstractmethod
    def forward(self, x1, x2, **kwargs):
        raise NotImplementedError()

    def __call__(self, x1=None, x2=None, **kwargs):
        if x1 is None:
            return self.forward(self.x1, self.x2)
        elif x2 is None:
            return self.forward(x1, self.x2)
        else:
            return self.forward(x1, x2, **kwargs)

    @staticmethod
    def toArray(*values):
        return np.array(values).reshape(np.shape(values))

    def __add__(self, other):
        raise NotImplementedError()

    def __radd__(self, other):
        raise NotImplementedError()

    def __mul__(self, other):
        raise NotImplementedError()

    def __rmul__(self, other):
        raise NotImplementedError()


class Foo():
    def say(self):
        return "foo"
